"""Shared helpers for the hybrid inference project.

Kept free of top-level torch / tensorflow imports: Phase 3 scripts run
TensorFlow and must not load torch in the same process.
"""
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import yaml

HI_ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


# ----------------------------------------------------------------------------
# Configuration, artifacts, manifest
# ----------------------------------------------------------------------------

def add_config_arg(parser):
    parser.add_argument("--config", default=str(HI_ROOT / "config" / "hybrid.yaml"))
    return parser


def _resolve(p):
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else HI_ROOT / p


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    cfg["repo"] = str(Path(cfg["repo"]))
    cfg["checkpoint"] = str(_resolve(cfg["checkpoint"])) if cfg.get("checkpoint") else None
    cfg["data"]["data_dir"] = str(_resolve(cfg["data"]["data_dir"]))
    return cfg


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def run_tag(cfg):
    """standin-s<seed> for the stand-in, ckpt-<sha8> for a real checkpoint."""
    if cfg["checkpoint"] is None:
        return f"standin-s{cfg['standin']['seed']}"
    return f"ckpt-{sha256_file(cfg['checkpoint'])[:8]}"


def art_dir(cfg, *parts, tag=None):
    d = HI_ROOT / "artifacts" / (tag or run_tag(cfg))
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(cfg):
    if cfg["checkpoint"] is not None:
        return Path(cfg["checkpoint"])
    return art_dir(cfg, "checkpoint") / "standin.ckpt"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def update_manifest(keys, data):
    """Set manifest[keys[0]][keys[1]]... = data (atomic rewrite)."""
    path = HI_ROOT / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    node = man
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = data
    write_json(path, man)


def fpga_name(cfg):
    """Name of the bitstream in use: artifacts/<tag>/fpga/<name>.{xclbin,json} (config fpga.name)."""
    return cfg.get("fpga", {}).get("name", "pair_head")


def fpga_meta_path(cfg):
    return art_dir(cfg, "fpga") / f"{fpga_name(cfg)}.json"


def variant_name(cfg, base):
    """Report / manifest key for results that depend on the bitstream: p4_fpga, p4_fpga_<name>, ..."""
    n = fpga_name(cfg)
    return base if n == "pair_head" else f"{base}_{n}"


def save_report(cfg, name, obj):
    path = art_dir(cfg, "reports") / f"{name}.json"
    write_json(path, obj)
    return path


class Gates:
    """Collects pass/fail checks; scripts exit non-zero when any gate fails."""

    def __init__(self):
        self.items = []

    def check(self, name, ok, **info):
        ok = bool(ok)
        self.items.append({"name": name, "ok": ok, **info})
        detail = ", ".join(f"{k}={_fmt(v)}" for k, v in info.items())
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return ok

    @property
    def ok(self):
        return all(i["ok"] for i in self.items)

    def summary(self):
        return {"passed": self.ok, "n_gates": len(self.items),
                "failed": [i["name"] for i in self.items if not i["ok"]], "gates": self.items}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}"
    return v


# ----------------------------------------------------------------------------
# Repository import (read-only) with stubs for training-only dependencies
# ----------------------------------------------------------------------------

STUBS_ACTIVE = {}


def _install_stubs():
    """pytorch_lightning and focal_loss are imported at the top of
    src/models/Transformer.py but only used for training. When they are not
    installed, register minimal stand-ins so Encoder / EncoderONNX import
    unchanged. The training paths are never used here."""
    import importlib.util
    import torch.nn as nn

    if "pytorch_lightning" not in sys.modules and importlib.util.find_spec("pytorch_lightning") is None:
        pl = types.ModuleType("pytorch_lightning")

        class LightningModule(nn.Module):
            def save_hyperparameters(self, *a, **k):
                pass

            def log(self, *a, **k):
                pass

        pl.LightningModule = LightningModule
        pl.seed_everything = lambda seed=None, **k: seed
        pl.__hi_stub__ = True
        sys.modules["pytorch_lightning"] = pl
        STUBS_ACTIVE["pytorch_lightning"] = True

    if "focal_loss" not in sys.modules and importlib.util.find_spec("focal_loss") is None:
        fl = types.ModuleType("focal_loss")
        flf = types.ModuleType("focal_loss.focal_loss")

        class FocalLoss(nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, *a, **k):
                raise RuntimeError("FocalLoss stub: inference only")

        flf.FocalLoss = FocalLoss
        fl.focal_loss = flf
        sys.modules["focal_loss"] = fl
        sys.modules["focal_loss.focal_loss"] = flf
        STUBS_ACTIVE["focal_loss"] = True


def import_repo(cfg):
    """Return the repo module src.models.Transformer (Encoder, EncoderONNX)."""
    _install_stubs()
    if cfg["repo"] not in sys.path:
        sys.path.insert(0, cfg["repo"])
    import src.models.Transformer as T
    return T


def repo_git(cfg, *args):
    """git on the (root-owned) repo as the current user. git 2.25 ignores
    `-c safe.directory`, so point HOME at a throwaway .gitconfig instead of
    touching the user's global config."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as home:
        Path(home, ".gitconfig").write_text("[safe]\n\tdirectory = *\n")
        r = subprocess.run(["git", "-C", cfg["repo"], *args], capture_output=True, text=True,
                           env={**os.environ, "HOME": home})
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def repo_clean(cfg):
    return repo_git(cfg, "status", "--porcelain") == ""


def import_metrics(cfg):
    if cfg["repo"] not in sys.path:
        sys.path.insert(0, cfg["repo"])
    import src.tools.support_functions_metrics as M
    return M


# ----------------------------------------------------------------------------
# Checkpoint / model
# ----------------------------------------------------------------------------

class _Placeholder(dict):
    """Stands in for pickled classes from packages not installed here."""

    def __init__(self, *a, **k):
        super().__init__()

    def __setstate__(self, state):
        self["__state__"] = state


class _LenientUnpickler(pickle.Unpickler):
    _missing = ("pytorch_lightning", "lightning", "lightning_fabric", "focal_loss", "wandb", "omegaconf")

    def find_class(self, module, name):
        if module.split(".")[0] in self._missing:
            try:
                return super().find_class(module, name)
            except Exception:
                return _Placeholder
        return super().find_class(module, name)


def _lenient_pickle_module():
    mod = types.ModuleType("hi_lenient_pickle")
    mod.__dict__.update({k: v for k, v in pickle.__dict__.items() if not k.startswith("__")})
    mod.Unpickler = _LenientUnpickler
    return mod


def load_checkpoint(path):
    """Load a Lightning checkpoint. Returns (state_dict without 'model.', hparams config, meta)."""
    import torch
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        lenient = False
    except (ModuleNotFoundError, AttributeError):
        ckpt = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_lenient_pickle_module())
        lenient = True
    sd_full = ckpt["state_dict"]
    # Same key mapping as src/ModelsToONNX/TransformerToOnnx.py: drop the 'model.' prefix
    sd = {k[len("model."):]: v for k, v in sd_full.items() if k.startswith("model.")}
    dropped = [k for k in sd_full if not k.startswith("model.")]
    hp = ckpt.get("hyper_parameters") or {}
    hp_cfg = hp.get("config") if isinstance(hp, dict) else None
    meta = {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "pytorch_lightning_version": ckpt.get("pytorch-lightning_version"),
        "dropped_keys": dropped,
        "lenient_unpickle": lenient,
        "standin": ckpt.get("hi_standin"),
        "has_hparams_config": hp_cfg is not None,
    }
    return sd, hp_cfg, meta


ARCH_KEYS = ("input_dim", "d_model", "num_head", "num_layers", "hidden_dim_classifier", "dropout", "time")


def model_config(cfg, hp_cfg=None):
    """Repo YAML, overridden by the checkpoint's stored config (architecture + time mode).
    Checkpoints without stored hyperparameters (e.g. the tutorial notebook's) take the
    architecture from `model_arch` in the hybrid config: num_head cannot be checked
    from the weights, so it must be stated, not guessed."""
    with open(Path(cfg["repo"]) / cfg["model_yaml"]) as f:
        mcfg = yaml.safe_load(f)
    diffs = {}
    source = None
    if hp_cfg and "model" in hp_cfg:
        override, source = hp_cfg["model"], "checkpoint hyper_parameters"
    elif cfg.get("model_arch"):
        override, source = cfg["model_arch"], "hybrid config model_arch"
    else:
        override = {}
    for k in ARCH_KEYS:
        if k in override and override[k] != mcfg["model"].get(k):
            diffs[k] = {"yaml": mcfg["model"].get(k), source: override[k]}
            mcfg["model"][k] = override[k]
    mcfg["model"]["arch_source"] = source
    mcfg["model"]["device"] = "cpu"
    mcfg["data"]["data_dir"] = cfg["data"]["data_dir"]
    return mcfg, diffs


def build_models(cfg, mcfg, sd):
    """Repo Encoder (reference, nn.TransformerEncoder fast path) and EncoderONNX, strict load."""
    T = import_repo(cfg)
    enc = T.Encoder(mcfg)
    enc.load_state_dict(sd, strict=True)
    enc.eval()
    onx = T.EncoderONNX(mcfg)
    onx.load_state_dict(sd, strict=True)
    onx.eval()
    return enc, onx


def load_model(cfg):
    """Everything later phases need: (mcfg, sd, meta, Encoder, EncoderONNX)."""
    path = checkpoint_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run p0_prepare.py first")
    sd, hp_cfg, meta = load_checkpoint(path)
    mcfg, diffs = model_config(cfg, hp_cfg)
    meta["config_diffs"] = diffs
    enc, onx = build_models(cfg, mcfg, sd)
    return mcfg, sd, meta, enc, onx


def make_standin(cfg, mcfg, calib_events):
    """Stand-in weights for the current architecture: every parameter drawn
    independently (fixed seed) so per-layer mix-ups, transposes or dropped
    biases cannot pass the equivalence checks; fc2 rescaled so that the
    calibration logits spread over all thresholds."""
    import torch
    T = import_repo(cfg)
    seed = cfg["standin"]["seed"]
    torch.manual_seed(seed)
    enc = T.Encoder(mcfg)
    with torch.no_grad():
        for name, p in enc.named_parameters():
            if "norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn_like(p))
            else:
                fan_in = p.shape[1] if p.dim() > 1 else p.shape[0]
                p.copy_(torch.randn_like(p) / fan_in ** 0.5)
    enc.eval()

    with torch.no_grad():
        # Calibrate the embedding to ~unit-scale outputs: the hit features are
        # not normalised (x, y up to +-30 mm), and an uncalibrated embedding
        # saturates the first attention softmax (one-hot rows), which would
        # make the attention checks trivially pass.
        xs = torch.from_numpy(np.concatenate([ev["features"] for ev in calib_events]))
        e = xs @ enc.embedding.weight.T
        enc.embedding.weight.div_(e.std(0, keepdim=True).T)
        enc.embedding.bias.copy_(-(e.mean(0) / e.std(0)) + 0.1 * torch.randn(e.shape[1]))

    # Calibrate fc2: std(l1 - l0) = logit_std, mean 0
    diffs = []
    with torch.no_grad():
        for ev in calib_events:
            x = torch.from_numpy(ev["features"])[None]
            n = x.shape[1]
            lg = enc(x, [ev["df"]], n)[0, : n * (n + 1) // 2]
            diffs.append((lg[:, 1] - lg[:, 0]).numpy())
        d = np.concatenate(diffs)
        scale = cfg["standin"]["logit_std"] / float(d.std())
        shift = float(d.mean()) * scale
        enc.fc2.weight.mul_(scale)
        enc.fc2.bias.mul_(scale)
        enc.fc2.bias[1] -= shift / 2
        enc.fc2.bias[0] += shift / 2

    sd = {"model." + k: v.detach().clone() for k, v in enc.state_dict().items()}
    return {
        "state_dict": sd,
        "hyper_parameters": {"config": mcfg},
        "hi_standin": {"seed": seed, "logit_std": cfg["standin"]["logit_std"],
                       "calib_pairs": int(d.size), "fc2_scale": scale, "fc2_shift": shift},
    }


# ----------------------------------------------------------------------------
# Pairs, softmax, comparison statistics
# ----------------------------------------------------------------------------

_PAIR_CACHE = {}


def pair_index(n):
    """[K,2] int64 pairs in combinations_with_replacement(range(n), 2) order."""
    if n not in _PAIR_CACHE:
        i, j = np.triu_indices(n)
        _PAIR_CACHE[n] = np.ascontiguousarray(np.stack([i, j], axis=1).astype(np.int64))
    return _PAIR_CACHE[n]


def pair_features(emb0, idx):
    """emb0 [N,D], idx [K,2] -> [K,2D] = concat(emb0[i], emb0[j])."""
    return np.ascontiguousarray(emb0[idx].reshape(idx.shape[0], -1), dtype=np.float32)


def assert_pair_order(cfg, n_max=150):
    M = import_metrics(cfg)
    for n in range(0, n_max + 1):
        ref = np.array(M.find_all_combinations(list(range(n))), dtype=np.int64).reshape(-1, 2)
        if not np.array_equal(ref, pair_index(n)):
            raise AssertionError(f"pair order mismatch at N={n}")
    return True


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# Edges for the track building: the repo's edge_dictionary_transformer keeps {(i, j): p1} for the pairs
# with p1 > threshold (float32 comparison), in pair order; nothing else of the probabilities is used.
EDGE_LOGIT_MARGIN = 1e-3   # logit units; float32 rounding of p1 near the threshold is ~1e-6 in logit units


def logit_threshold(thr):
    """p1 = softmax(l)[1] = sigmoid(l1 - l0), so p1 > thr  <=>  l1 - l0 > log(thr / (1 - thr))."""
    return float(np.log(thr / (1.0 - thr)))


def edges_from_probs(p1, thr):
    """(rows, scores) of the pairs with p1 > thr: the selection of edge_dictionary_transformer."""
    p1 = np.asarray(p1, np.float32)
    sel = np.flatnonzero(p1 > np.float32(thr))
    return sel, p1[sel]


def edges_from_logits(lg, thr):
    """Same result as edges_from_probs(softmax(lg)[:, 1], thr), bit for bit, without the softmax over all
    K pairs: only rows with l1 - l0 above logit(thr) - margin can pass; the softmax and the exact
    p1 > thr test are applied to those candidate rows only (the softmax is row-wise)."""
    lg = np.asarray(lg, np.float32)
    l0, l1 = lg[:, 0], lg[:, 1]
    cand = np.flatnonzero(l1 - l0 > logit_threshold(thr) - EDGE_LOGIT_MARGIN)
    # softmax(...)[:, 1] of the candidate rows written column-wise (numpy reductions over a 2-wide axis are
    # slow): the same float32 operations in the same order (max, x - max, exp, e0 + e1, e1 / sum)
    a, b = l0[cand], l1[cand]
    m = np.maximum(a, b)
    e0, e1 = np.exp(a - m), np.exp(b - m)
    p1 = e1 / (e0 + e1)
    keep = p1 > np.float32(thr)
    return cand[keep], p1[keep]


def edge_dict(idx, sel, scores):
    """{(i, j): score} in pair order, as edge_dictionary_transformer builds it."""
    return {(int(i), int(j)): float(s) for (i, j), s in zip(idx[sel], scores)}


# On-card path (kernel hls_ext/pairedges_kernel.cpp): the card returns edge candidates packed as uint64
#   [15:0] i, [31:16] j, [47:32] raw(l0), [63:48] raw(l1)   (raw = logit * 2^frac, signed)
CAND_PER_WORD = 8   # candidates per 512-bit word; each event's block starts on a word


def candidate_thr_raw(thr, frac):
    """Card-side candidate test raw(l1) - raw(l0) > thr_raw: a superset of the edges (margin as edges_from_logits)."""
    return int(np.floor((logit_threshold(thr) - EDGE_LOGIT_MARGIN) * 2 ** frac))


def split_candidates(out_words, counts):
    """Per-event candidate arrays from the packed card output."""
    res, pos = [], 0
    for c in counts:
        res.append(out_words[pos:pos + int(c)])
        pos += -(-int(c) // CAND_PER_WORD) * CAND_PER_WORD
    return res


def unpack_candidates(words, frac):
    """uint64 candidates -> (i, j, logits [E, 2] float32); logits = raw * 2^-frac (exact in float32)."""
    w = np.asarray(words, np.uint64)
    i = (w & np.uint64(0xFFFF)).astype(np.int64)
    j = ((w >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.int64)
    l0 = ((w >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.int16)
    l1 = ((w >> np.uint64(48)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.int16)
    return i, j, np.stack([l0, l1], 1).astype(np.float32) * np.float32(2.0 ** -frac)


def edges_from_unpacked(rows, l0, l1, thr):
    """edges_from_candidates on candidates already unpacked by the driver (rows, float32 logits): the same
    float32 operations, so the same rows and bit-identical p1."""
    m = np.maximum(l0, l1)
    e0, e1 = np.exp(l0 - m), np.exp(l1 - m)
    p1 = e1 / (e0 + e1)
    keep = p1 > np.float32(thr)
    return rows[keep], p1[keep]


def edges_from_candidates(words, n, frac, thr):
    """Host side of the on-card path: the exact softmax + p1 > thr of edges_from_logits on the candidates.
    Returns (rows in pair_index order, p1, (i, j))."""
    i, j, lg = unpack_candidates(words, frac)
    a, b = lg[:, 0], lg[:, 1]
    m = np.maximum(a, b)
    e0, e1 = np.exp(a - m), np.exp(b - m)
    p1 = e1 / (e0 + e1)
    keep = p1 > np.float32(thr)
    i, j = i[keep], j[keep]
    return i * n - i * (i - 1) // 2 + (j - i), p1[keep], np.stack([i, j], 1)


def err_stats(a, b):
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).ravel()
    if d.size == 0:
        return {"max": 0.0, "mean": 0.0, "p99": 0.0, "n": 0}
    return {"max": float(d.max()), "mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "n": int(d.size)}


def flip_stats(p_ref, p_new, thresholds, band):
    """Decision changes of 'p1 > t' (the repo's edge rule) between two prob vectors.
    flips_outside_band counts flips where the reference is at least `band` away from t."""
    p_ref = np.asarray(p_ref, np.float64).ravel()
    p_new = np.asarray(p_new, np.float64).ravel()
    out = {}
    for t in thresholds:
        flip = (p_ref > t) != (p_new > t)
        out[str(t)] = {"flips": int(flip.sum()),
                       "flips_outside_band": int((flip & (np.abs(p_ref - t) >= band)).sum())}
    return out


def thresholds(cfg, mcfg):
    ts = sorted(set(float(t) for t in mcfg.get("metrics", {}).get("threshold", [])) | {float(cfg["operating_threshold"])})
    return ts


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------

def iter_events(cfg, mcfg, split, limit=None):
    """Events of a split via the repo's TransformerDataset (same feature
    selection, incl. model.time), in numeric event-id order.
    Yields dict(name, features f32 [N,4], df, gt [K,2])."""
    if cfg["repo"] not in sys.path:
        sys.path.insert(0, cfg["repo"])
    from src.Dataloader.Transformer.dataset import TransformerDataset
    ds = TransformerDataset(mcfg, split)
    ds.features = sorted(ds.features, key=lambda f: int(f.split(".")[0].split("_")[-1]))
    n = len(ds) if limit is None else min(limit, len(ds))
    for k in range(n):
        features, gt, df, name = ds[k]
        yield {"name": name, "features": features.numpy().astype(np.float32), "df": df, "gt": gt.numpy()}


def load_events(cfg, mcfg, split, limit=None):
    return list(iter_events(cfg, mcfg, split, limit if limit is not None else cfg["data"].get("limit")))


def _synthetic_event(rng, n_min, n_max):
    import pandas as pd
    rows = []
    kin = 1
    n_tracks = int(rng.integers(1, 13))
    t_tracks = []
    for _ in range(n_tracks):
        x0, y0 = rng.uniform(-30, 30, 2)
        dx, dy = rng.normal(0, 0.5, 2)
        t0 = rng.uniform(-10, 10)
        t_tracks.append(t0)
        for s in range(4):
            if rng.random() < 0.95:
                x = float(np.clip(x0 + dx * s + rng.normal(0, 0.1), -30, 30))
                y = float(np.clip(y0 + dy * s + rng.normal(0, 0.1), -30, 30))
                t = t0 + rng.normal(0, 0.15)
                rows.append([x, y, t, s, kin])
                if rng.random() < 0.1:   # split cluster: neighbour pixel, same time
                    rows.append([float(np.clip(x + 0.3, -30, 30)), y, t, s, kin])
        kin += 1
    for _ in range(int(rng.integers(0, 13))):   # noise hits, unique kinepart each
        rows.append([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-10, 10), int(rng.integers(0, 4)), kin])
        kin += 1
    rows = [rows[k] for k in rng.permutation(len(rows))]
    if len(rows) > n_max:
        rows = rows[:n_max]
    if len(rows) < max(n_min, 3):
        return None
    df = pd.DataFrame(rows, columns=["cl_x", "cl_y", "cl_time", "GTKstation", "cl_kinepart"])
    st = df["GTKstation"]
    if not ((st == 3).any() and (st == 2).any() and ((st == 0).any() or (st == 1).any())):
        return None
    df["event_number"] = 0.0
    df["cl_burst"] = 0.0
    df["KTAGTime"] = t_tracks[0] + rng.normal(0, 0.1)
    df["parent_particle"] = -1.0
    df = df[["cl_x", "cl_y", "cl_time", "GTKstation", "event_number", "cl_burst",
             "cl_kinepart", "KTAGTime", "parent_particle"]]
    return df.astype(np.float16).astype(np.float64)   # pre-processing stores float16 values


def _synthetic_gt(df):
    """Edges between hits of the same kinepart on consecutive *present* stations
    (approximation of the Monconi rule), as [K,2] one-hot ([0,1] = true edge)."""
    from itertools import product
    true = set()
    for _, g in df.groupby("cl_kinepart"):
        by_st = [list(g.index[g["GTKstation"] == s]) for s in range(4)]
        present = [s for s in range(4) if by_st[s]]
        for a, b in zip(present[:-1], present[1:]):
            for i, j in product(by_st[a], by_st[b]):
                true.add((min(i, j), max(i, j)))
    idx = pair_index(len(df))
    lab = np.array([(int(i), int(j)) in true for i, j in idx])
    gt = np.zeros((len(idx), 2), np.float32)
    gt[lab, 1] = 1
    gt[~lab, 0] = 1
    return gt


def make_synthetic_dataset(out_dir, seed, n_per_split, n_min, n_max):
    """Synthetic events in the real pre-processed on-disk format."""
    import torch
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    offsets = {"train": 0, "validation": 100000, "test": 200000}
    for split, n_events in n_per_split.items():
        for sub in ("features", "gt_tensors", "dataframe"):
            (out_dir / sub / split).mkdir(parents=True, exist_ok=True)
        made = 0
        while made < n_events:
            df = _synthetic_event(rng, n_min, n_max)
            if df is None:
                continue
            name = str(offsets[split] + made)
            df.to_csv(out_dir / "dataframe" / split / f"df_{name}.csv", index=False)
            feats = torch.tensor(df[["cl_x", "cl_y", "GTKstation", "cl_time", "KTAGTime"]].values, dtype=torch.float32)
            torch.save(feats, out_dir / "features" / split / f"df_{name}.pt")
            torch.save(torch.from_numpy(_synthetic_gt(df)), out_dir / "gt_tensors" / split / f"gt_tensors_{name}.pt")
            made += 1


# ----------------------------------------------------------------------------
# ONNX helpers
# ----------------------------------------------------------------------------

def mlp_layers_from_onnx(model):
    """Dense layers of a MatMul(+Add) / Gemm chain, in graph order, as
    [{'W': [in,out], 'b': [out], 'W_name', 'b_name'}]. Weights stay exactly as
    stored (no re-rounding)."""
    from onnx import numpy_helper
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    layers = []
    for n in model.graph.node:
        if n.op_type == "MatMul":
            w = [x for x in n.input if x in inits]
            if len(w) == 1:
                layers.append({"W": inits[w[0]], "W_name": w[0], "b": None, "b_name": None, "out": n.output[0]})
        elif n.op_type == "Gemm":
            attrs = {a.name: a for a in n.attribute}
            alpha = attrs["alpha"].f if "alpha" in attrs else 1.0
            beta = attrs["beta"].f if "beta" in attrs else 1.0
            if alpha != 1.0 or beta != 1.0 or ("transA" in attrs and attrs["transA"].i):
                raise ValueError(f"unsupported Gemm attributes in {n.name}")
            w = inits[n.input[1]]
            if "transB" in attrs and attrs["transB"].i:
                w = w.T
            layers.append({"W": w, "W_name": n.input[1], "b": inits.get(n.input[2]) if len(n.input) > 2 else None,
                           "b_name": n.input[2] if len(n.input) > 2 else None, "out": n.output[0]})
        elif n.op_type == "Add" and layers and layers[-1]["b"] is None and layers[-1]["out"] in n.input:
            b = [x for x in n.input if x in inits]
            if len(b) == 1:
                layers[-1]["b"], layers[-1]["b_name"] = inits[b[0]], b[0]
                layers[-1]["out"] = n.output[0]
    return layers


# ----------------------------------------------------------------------------
# ONNX Runtime
# ----------------------------------------------------------------------------

def ort_session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if isinstance(path, (bytes, bytearray)):
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
