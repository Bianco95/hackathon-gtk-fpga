"""Phase 5: application-level hybrid inference API.

    pipe = HybridPipeline(encoder_onnx, make_pair_head("fpga", cfg))
    res = pipe.infer(features)          # features [N,4] float32, one event
    res.probs                           # [1,K,2], rows in combinations_with_replacement(range(N), 2) order
    res.pairs                           # [K,2] the (i, j) of every output row

The pair head is interchangeable: every backend implements
classify_pairs(pair_features [K,64] float32) -> logits [K,2] (see pairhead_backends.py).
Post-processing reuses src/tools/support_functions_metrics.py of the repo unchanged.
"""
import json
import time
from dataclasses import dataclass, field

import numpy as np

import hi_common as H
from pairhead_backends import PairEdgesCSim, PairEdgesFPGA, PairHeadCPU, PairHeadCSim, PairHeadFPGA


def make_pair_head(kind, cfg, threads=1, project_dir=None):
    """kind: 'cpu' (ORT on pair_head.onnx), 'fpga' (U55C, fpga/<fpga.name>.json), 'csim' (hls4ml C-sim)."""
    if kind == "cpu":
        return PairHeadCPU(H.art_dir(cfg, "onnx") / "pair_head.onnx", threads)
    if kind == "fpga":
        f = cfg["fpga"]
        return PairHeadFPGA.from_meta(H.fpga_meta_path(cfg), device=f["device"],
                                      max_chunk=f["max_chunk_pairs"], timeout_ms=f["timeout_ms"])
    if kind == "csim":
        if project_dir is None:
            meta = H.fpga_meta_path(cfg)
            if meta.exists():
                project_dir = json.loads(meta.read_text())["project"]
            else:   # before the bitstream exists: the selected scan project (same numerics)
                sel = json.loads((H.art_dir(cfg, "hls", "scan") / "precision_selected.json").read_text())
                project_dir = H.art_dir(cfg, "hls", "scan") / sel["name"]
        w = np.load(H.art_dir(cfg, "keras") / "pair_head_weights.npz")
        return PairHeadCSim(project_dir, w["fc1_W"].shape[0], w["fc2_W"].shape[1], cfg["hls"]["project_name"])
    raise ValueError(f"unknown pair head '{kind}'")


@dataclass
class EventResult:
    probs: np.ndarray            # [1,K,2] softmax probabilities (shape expected by edge_dictionary_transformer);
                                 # None with post="edges"
    logits: np.ndarray           # [K,2]
    pairs: np.ndarray            # [K,2] (i, j) of each row; None for OnCardPipeline (rows are in pair_index order)
    timings: dict = field(default_factory=dict)   # microseconds per stage
    edges: tuple = None          # post="edges": (rows [E], p1 [E]) of the pairs with p1 > threshold

    @property
    def n_pairs(self):
        return self.pairs.shape[0]


class HybridPipeline:
    """post="softmax": probabilities of all K pairs (host softmax).
    post="edges": only the edges the track building uses, (rows, p1) with p1 > threshold, from the logits
    (H.edges_from_logits: bit-identical to thresholding the softmax, no softmax over all K pairs)."""

    def __init__(self, encoder_onnx, head, ort_threads=1, post="softmax", threshold=None):
        if post not in ("softmax", "edges"):
            raise ValueError(f"unknown post-processing '{post}'")
        if post == "edges" and threshold is None:
            raise ValueError('post="edges" needs the threshold')
        self.enc = H.ort_session(encoder_onnx, ort_threads)
        self.head = head
        self.n_in = head.n_in
        self.d_model = head.n_in // 2
        self.post, self.threshold = post, threshold

    def _post(self, lg):
        """(probs [1,K,2] or None, edges or None)"""
        if self.post == "edges":
            return None, H.edges_from_logits(lg, self.threshold)
        return H.softmax(lg)[None], None

    def _check(self, features):
        x = np.ascontiguousarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected one event [N,F], got {x.shape}")
        return x

    def _embed(self, x, t):
        a = time.perf_counter_ns()
        emb = self.enc.run(None, {"data": x})[0]          # [1,N,D]
        b = time.perf_counter_ns()
        idx = H.pair_index(x.shape[0])
        pf = H.pair_features(emb[0], idx)                 # [K,2D], row r = concat(emb[i_r], emb[j_r])
        c = time.perf_counter_ns()
        t["encoder_us"] = t.get("encoder_us", 0.0) + (b - a) / 1e3
        t["pairgen_us"] = t.get("pairgen_us", 0.0) + (c - b) / 1e3
        return idx, pf

    def infer(self, features):
        x = self._check(features)
        t = {}
        if x.shape[0] == 0:
            lg = np.empty((0, 2), np.float32)
            p, ed = self._post(lg)
            return EventResult(p, lg, np.empty((0, 2), np.int64), t, ed)
        idx, pf = self._embed(x, t)
        a = time.perf_counter_ns()
        lg = self.head.classify_pairs(pf)
        b = time.perf_counter_ns()
        p, ed = self._post(lg)
        c = time.perf_counter_ns()
        t["head_us"] = (b - a) / 1e3
        t["softmax_us" if self.post == "softmax" else "edges_us"] = (c - b) / 1e3
        if getattr(self.head, "last_timing", None):
            t.update({f"head_{k}": v for k, v in self.head.last_timing.items()})
        return EventResult(p, lg, idx, t, ed)

    def infer_batch(self, events):
        """Several events, one pair-head call over their concatenated pairs (amortises the
        per-call cost of the FPGA), split back by offsets. Results equal per-event infer()."""
        xs = [self._check(e) for e in events]
        t = {}
        parts, idxs = [], []
        for x in xs:
            if x.shape[0] == 0:
                idxs.append(np.empty((0, 2), np.int64))
                parts.append(np.empty((0, self.n_in), np.float32))
                continue
            idx, pf = self._embed(x, t)
            idxs.append(idx)
            parts.append(pf)
        offs = np.cumsum([0] + [p.shape[0] for p in parts])
        a = time.perf_counter_ns()
        allpf = np.concatenate(parts) if offs[-1] else np.empty((0, self.n_in), np.float32)
        b = time.perf_counter_ns()
        lg = self.head.classify_pairs(allpf)
        c = time.perf_counter_ns()
        if self.post == "edges":
            out = [EventResult(None, lg[s:e], idx, {}, H.edges_from_logits(lg[s:e], self.threshold))
                   for s, e, idx in zip(offs[:-1], offs[1:], idxs)]
        else:
            p = H.softmax(lg) if lg.shape[0] else np.empty((0, 2), np.float32)
            out = [EventResult(p[s:e][None], lg[s:e], idx, {}) for s, e, idx in zip(offs[:-1], offs[1:], idxs)]
        d = time.perf_counter_ns()
        t.update({"concat_us": (b - a) / 1e3, "head_us": (c - b) / 1e3,
                  "softmax_us" if self.post == "softmax" else "edges_us": (d - c) / 1e3})
        if getattr(self.head, "last_timing", None):
            t.update({f"head_{k}": v for k, v in self.head.last_timing.items()})
        return out, t


def make_card(kind, cfg, kernel_dir=None, mode=1):
    """Pair building + MLP + candidate filter on the card: 'fpga' (fpga/pairedges.json; mode 1 = start /
    completion through the control register, 0 = through XRT) or 'csim' (C simulation of the kernel in
    kernel_dir, default the one of the installed bitstream)."""
    meta_path = H.art_dir(cfg, "fpga") / "pairedges.json"
    if kind == "fpga":
        f = cfg["fpga"]
        return PairEdgesFPGA.from_meta(meta_path, device=f["device"], timeout_ms=f["timeout_ms"], mode=mode)
    if kind == "csim":
        kdir = kernel_dir or json.loads(meta_path.read_text())["kernel_dir"]
        return PairEdgesCSim(kdir)
    raise ValueError(f"unknown card '{kind}'")


class OnCardPipeline:
    """encoder.onnx on the host -> embeddings -> card (pair building, MLP, candidate filter; the driver
    unpacks the candidates) -> host: exact softmax + threshold on the candidates (H.edges_from_unpacked).
    Output: the edges, as HybridPipeline(post="edges"): EventResult.edges = (rows in pair_index order, p1)."""

    def __init__(self, encoder_onnx, card, threshold, frac_bits, ort_threads=1, timings=True):
        self.timings = timings and hasattr(card, "timing")   # driver breakdown in EventResult.timings
        self.enc = H.ort_session(encoder_onnx, ort_threads)
        self.card = card
        self.threshold, self.frac = threshold, frac_bits
        self.thr_raw = H.candidate_thr_raw(threshold, frac_bits)
        self.post = "edges"

    def _empty(self, t):
        return EventResult(None, None, None, t, (np.empty(0, np.int64), np.empty(0, np.float32)))

    def infer(self, features):
        x = np.ascontiguousarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"expected one event [N,F], got {x.shape}")
        t = {}
        if x.shape[0] == 0:
            return self._empty(t)
        a = time.perf_counter_ns()
        emb = self.enc.run(None, {"data": x})[0][0]
        b = time.perf_counter_ns()
        rows, l0, l1, _ = self.card.run_unpacked([emb], self.thr_raw, self.frac)
        c = time.perf_counter_ns()
        edges = H.edges_from_unpacked(rows, l0, l1, self.threshold)
        d = time.perf_counter_ns()
        t.update({"encoder_us": (b - a) / 1e3, "card_us": (c - b) / 1e3, "edges_us": (d - c) / 1e3})
        if self.timings:
            t.update({f"card_{k}": v for k, v in self.card.timing.items()})
        return EventResult(None, None, None, t, edges)

    def infer_batch(self, events):
        """Several events, one card call; results equal per-event infer()."""
        xs = [np.ascontiguousarray(e, dtype=np.float32) for e in events]
        t = {}
        a = time.perf_counter_ns()
        embs = [self.enc.run(None, {"data": x})[0][0] for x in xs if x.shape[0]]
        b = time.perf_counter_ns()
        rows, l0, l1, cnt = self.card.run_unpacked(embs, self.thr_raw, self.frac) if embs else (None, None, None, [])
        c = time.perf_counter_ns()
        out = []
        if embs:   # one softmax over all candidates (row-wise), then split by event
            m = np.maximum(l0, l1)
            e0, e1 = np.exp(l0 - m), np.exp(l1 - m)
            p1 = e1 / (e0 + e1)
            keep = p1 > np.float32(self.threshold)
            offs = np.concatenate([[0], np.cumsum(cnt)])
        k = 0
        for x in xs:
            if x.shape[0] == 0:
                out.append(self._empty({}))
                continue
            s_, e_ = offs[k], offs[k + 1]
            kk = keep[s_:e_]
            out.append(EventResult(None, None, None, {}, (rows[s_:e_][kk], p1[s_:e_][kk])))
            k += 1
        d = time.perf_counter_ns()
        t.update({"encoder_us": (b - a) / 1e3, "card_us": (c - b) / 1e3, "edges_us": (d - c) / 1e3})
        if self.timings and embs:
            t.update({f"card_{k}": v for k, v in self.card.timing.items()})
        return out, t


# ---------------------------------------------------------------------------
# Track reconstruction (Framework flow of the repo, model call replaced)
# ---------------------------------------------------------------------------

def cluster_event(M, df, features):
    """cluster_hits_by_distance_and_time on one event; returns (df_c, features_c [Nc,4] f32, old_to_new)."""
    import torch
    df_c, f_c, old_to_new = M.cluster_hits_by_distance_and_time(df, torch.from_numpy(features)[None])
    return df_c, f_c[0].numpy().astype(np.float32), old_to_new


def tracks_from_probs(M, probs, df_c, old_to_new, df, threshold):
    """Same steps as compute_metrics_Transformer_Framework after the model call."""
    import torch
    if len(df_c) < 2:   # edge_dictionary_transformer squeezes the scores: needs K >= 2
        return _tracks(M, None, df_c, old_to_new, df)
    return _tracks(M, M.edge_dictionary_transformer(torch.from_numpy(probs), threshold, df_c), df_c, old_to_new, df)


def tracks_from_edges(M, pairs, edges, df_c, old_to_new, df):
    """Same as tracks_from_probs, from the (rows, p1) edges of post="edges" (same edge dictionary)."""
    if len(df_c) < 2:
        return _tracks(M, None, df_c, old_to_new, df)
    return _tracks(M, H.edge_dict(pairs, *edges), df_c, old_to_new, df)


def _tracks(M, ed, df_c, old_to_new, df):
    gt = M.create_gt_tracks_Framework(df)
    tracks = []
    if ed is not None:
        pred = M.make_pred_tracks(ed, df_c)
        tracks = [M.inverse_cluster_mapping(t, old_to_new) for t in M.filter_tracks_Framework(pred, df_c)]
    tp, fp, fn = M.tp_fp_fn(tracks, gt)
    return {"tp": tp, "fp": fp, "fn": fn, "tracks": sorted(tuple(sorted(int(i) for i in t)) for t in tracks)}


def run_framework_event(pipe, M, df, features, threshold):
    df_c, f_c, o2n = cluster_event(M, df, features)
    res = pipe.infer(f_c)
    if res.edges is not None:
        pairs = res.pairs if res.pairs is not None else H.pair_index(f_c.shape[0])
        return res, tracks_from_edges(M, pairs, res.edges, df_c, o2n, df)
    return res, tracks_from_probs(M, res.probs, df_c, o2n, df, threshold)
