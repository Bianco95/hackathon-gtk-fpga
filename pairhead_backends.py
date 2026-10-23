"""Interchangeable pair-head backends: classify_pairs(pair_features [K,n_in]) -> logits [K,n_out].

PairHeadCPU  : ONNX Runtime on pair_head.onnx
PairHeadFPGA : U55C kernel through fpga_host/build/libpairhead_xrt.so (XRT native API, ctypes)
PairHeadCSim : hls4ml C simulation library (firmware/<project>-<stamp>.so), validation only

On-card pair building (kernel hls_ext/pairedges_kernel.cpp): run(embeddings list, thr_raw) ->
(per-event uint64 edge candidates, timing); see hi_common.edges_from_candidates.
PairEdgesFPGA: U55C kernel "pairedges" (pe_* in libpairhead_xrt.so)
PairEdgesCSim: C simulation of the same kernel (libpairedges_csim.so)
"""
import ctypes
import glob
import json
from pathlib import Path

import numpy as np

HI_ROOT = Path(__file__).resolve().parent
LIB = HI_ROOT / "fpga_host" / "build" / "libpairhead_xrt.so"


def _as_input(x, n_in):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != n_in:
        raise ValueError(f"expected [K,{n_in}] pair features, got {x.shape}")
    return x


class PairHeadCPU:
    kind = "cpu"

    def __init__(self, onnx_path, threads=1):
        import hi_common as H
        self.sess = H.ort_session(onnx_path, threads)
        i, o = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        self.input_name = i.name
        self.n_in = i.shape[1]
        self.n_out = o.shape[1]
        self.last_timing = {}

    def classify_pairs(self, pair_features):
        x = _as_input(pair_features, self.n_in)
        if x.shape[0] == 0:
            return np.empty((0, self.n_out), np.float32)
        return self.sess.run(None, {self.input_name: x})[0]


class _Timing(ctypes.Structure):
    _fields_ = [("copy_in_us", ctypes.c_double), ("h2d_us", ctypes.c_double), ("kernel_us", ctypes.c_double),
                ("d2h_us", ctypes.c_double), ("copy_out_us", ctypes.c_double), ("total_us", ctypes.c_double),
                ("n_chunks", ctypes.c_uint32)]


def _load_lib():
    lib = ctypes.CDLL(str(LIB))
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.ph_open.restype = ctypes.c_void_p
    lib.ph_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
                            ctypes.c_char_p]
    lib.ph_open_cus.restype = ctypes.c_void_p
    lib.ph_open_cus.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
                                ctypes.c_char_p, ctypes.c_uint32]
    lib.ph_classify.restype = ctypes.c_int
    lib.ph_classify.argtypes = [ctypes.c_void_p, f32p, ctypes.c_uint64, f32p, ctypes.c_uint32, ctypes.POINTER(_Timing)]
    lib.ph_classify_inplace.restype = ctypes.c_int
    lib.ph_classify_inplace.argtypes = [ctypes.c_void_p, ctypes.c_uint64, f32p, ctypes.c_uint32, ctypes.POINTER(_Timing)]
    lib.ph_input_view.restype = f32p
    lib.ph_input_view.argtypes = [ctypes.c_void_p]
    lib.ph_info.restype = ctypes.c_char_p
    lib.ph_info.argtypes = [ctypes.c_void_p]
    lib.ph_last_error.restype = ctypes.c_char_p
    lib.ph_last_error.argtypes = [ctypes.c_void_p]
    lib.ph_close.argtypes = [ctypes.c_void_p]
    return lib


class PairHeadFPGA:
    """One open handle per (xclbin, device): the bitstream stays loaded and the
    device buffers are reused across calls. n_cu > 1 drives several copies of the
    kernel (compute units) in parallel, each on its own HBM bank."""
    kind = "fpga"
    _instances = {}

    def __new__(cls, xclbin, n_in, n_out, device="0", max_chunk=262144, timeout_ms=10000, kernel="krnl_rtl", n_cu=1):
        key = (str(Path(xclbin).resolve()), str(device))
        if key not in cls._instances:
            obj = super().__new__(cls)
            obj._init(xclbin, n_in, n_out, device, max_chunk, timeout_ms, kernel, n_cu)
            cls._instances[key] = obj
        inst = cls._instances[key]
        if (inst.n_in, inst.n_out) != (n_in, n_out):
            raise ValueError(f"already open with n_in/n_out {inst.n_in}/{inst.n_out}")
        return inst

    @classmethod
    def from_meta(cls, meta_json, device=None, max_chunk=262144, timeout_ms=10000):
        m = json.loads(Path(meta_json).read_text())
        xclbin = Path(meta_json).with_name(m["xclbin"])
        return cls(xclbin, m["n_in"], m["n_out"], device or m.get("device", "0"), max_chunk, timeout_ms,
                   n_cu=int(m.get("n_cu", 1)))

    def _init(self, xclbin, n_in, n_out, device, max_chunk, timeout_ms, kernel, n_cu):
        self.lib = _load_lib()
        self.n_in, self.n_out, self.max_chunk, self.timeout_ms = int(n_in), int(n_out), int(max_chunk), int(timeout_ms)
        self.n_cu = int(n_cu)
        self.h = self.lib.ph_open_cus(str(xclbin).encode(), str(device).encode(), self.n_in, self.n_out,
                                      self.max_chunk, kernel.encode(), self.n_cu)
        if not self.h:
            raise RuntimeError(f"ph_open failed: {self.lib.ph_last_error(None).decode()}")
        self.info = json.loads(self.lib.ph_info(self.h).decode())
        self.last_timing = {}
        self._t = _Timing()

    def _timing(self):
        t = self._t
        self.last_timing = {f: getattr(t, f) for f, _ in _Timing._fields_}
        return self.last_timing

    def classify_pairs(self, pair_features):
        x = _as_input(pair_features, self.n_in)
        k = x.shape[0]
        out = np.empty((k, self.n_out), np.float32)
        if k == 0:
            self.last_timing = {}
            return out
        f32p = ctypes.POINTER(ctypes.c_float)
        rc = self.lib.ph_classify(self.h, x.ctypes.data_as(f32p), k, out.ctypes.data_as(f32p), self.timeout_ms,
                                  ctypes.byref(self._t))
        if rc != 0:
            raise RuntimeError(f"ph_classify failed: {self.lib.ph_last_error(self.h).decode()}")
        self._timing()
        return out

    def input_view(self):
        """Mapped input buffer [max_chunk, n_in] for zero-copy use with classify_inplace."""
        p = self.lib.ph_input_view(self.h)
        return np.ctypeslib.as_array(p, shape=(self.max_chunk, self.n_in))

    def classify_inplace(self, k):
        out = np.empty((k, self.n_out), np.float32)
        f32p = ctypes.POINTER(ctypes.c_float)
        rc = self.lib.ph_classify_inplace(self.h, k, out.ctypes.data_as(f32p), self.timeout_ms, ctypes.byref(self._t))
        if rc != 0:
            raise RuntimeError(f"ph_classify_inplace failed: {self.lib.ph_last_error(self.h).decode()}")
        self._timing()
        return out


class PairHeadCSim:
    """hls4ml C-simulation library of a generated project. The fork's bridge
    calls <project>_axi, i.e. the same float <-> fixed conversion as the kernel."""
    kind = "csim"

    def __init__(self, project_dir, n_in, n_out, project_name="myproject"):
        libs = sorted(glob.glob(str(Path(project_dir) / "firmware" / f"{project_name}-*.so")),
                      key=lambda p: Path(p).stat().st_mtime)
        if not libs:
            raise FileNotFoundError(f"no C-sim library in {project_dir}/firmware")
        self.lib = ctypes.CDLL(libs[-1])
        self.fn = getattr(self.lib, f"{project_name}_float")
        f32p = ctypes.POINTER(ctypes.c_float)
        self.fn.argtypes = [f32p, f32p]
        self.fn.restype = None
        self.n_in, self.n_out = n_in, n_out
        self.path = libs[-1]

    def classify_pairs(self, pair_features):
        x = _as_input(pair_features, self.n_in)
        out = np.empty((x.shape[0], self.n_out), np.float32)
        f32p = ctypes.POINTER(ctypes.c_float)
        for i in range(x.shape[0]):
            self.fn(x[i].ctypes.data_as(f32p), out[i].ctypes.data_as(f32p))
        return out


def _pack_embeddings(embs, d):
    n = np.array([e.shape[0] for e in embs], np.int32)
    if any(e.ndim != 2 or e.shape[1] != d for e in embs):
        raise ValueError(f"expected [N,{d}] embeddings")
    x = np.ascontiguousarray(np.concatenate(embs) if len(embs) else np.empty((0, d), np.float32), np.float32)
    k_words = int(sum(-(-(int(v) * (int(v) + 1) // 2) // 8) for v in n))
    return n, x, k_words * 8


class PairEdgesCSim:
    kind = "csim-edges"

    def __init__(self, kernel_dir, d=32):
        self.lib = ctypes.CDLL(str(Path(kernel_dir) / "libpairedges_csim.so"))
        f = self.lib.pairedges_csim
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        f.restype = ctypes.c_int
        self.f, self.d = f, d

    def run(self, embs, thr_raw):
        import hi_common as H
        n, x, cap = _pack_embeddings(embs, self.d)
        out = np.zeros(cap + 8, np.uint64)
        cnt = np.zeros(len(embs), np.int32)
        rc = self.f(x.ctypes.data, n.ctypes.data, len(embs), int(thr_raw), out.ctypes.data, cnt.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"pairedges_csim rc={rc}")
        return H.split_candidates(out, cnt), {}

    def run_unpacked(self, embs, thr_raw, frac):
        """Same result as PairEdgesFPGA.run_unpacked (unpacking done here in numpy)."""
        import hi_common as H
        cands, t = self.run(embs, thr_raw)
        rows, l0, l1 = [], [], []
        for c, e in zip(cands, embs):
            i, j, lg = H.unpack_candidates(c, frac)
            n = e.shape[0]
            rows.append((i * n - i * (i - 1) // 2 + (j - i)).astype(np.int32))
            l0.append(np.ascontiguousarray(lg[:, 0]))
            l1.append(np.ascontiguousarray(lg[:, 1]))
        cat = (lambda xs, dt: np.concatenate(xs) if xs else np.empty(0, dt))
        return cat(rows, np.int32), cat(l0, np.float32), cat(l1, np.float32), np.array([c.size for c in cands], np.int32)


class _PeTiming(ctypes.Structure):
    _fields_ = [("copy_in_us", ctypes.c_double), ("h2d_us", ctypes.c_double), ("kernel_us", ctypes.c_double),
                ("d2h_us", ctypes.c_double), ("copy_out_us", ctypes.c_double), ("total_us", ctypes.c_double),
                ("n_cand", ctypes.c_uint64)]


class PairEdgesFPGA:
    """U55C kernel 'pairedges': one call per list of events (embeddings in, candidates out)."""
    kind = "fpga-edges"
    _instances = {}

    def __init__(self, xclbin, d=32, device="0", max_hits=1 << 16, max_events=1024, max_cand=1 << 23,
                 timeout_ms=10000, kernel="pairedges", mode=1):
        """mode 1: start / completion through the kernel's control register (busy poll); 0: through XRT."""
        self.lib = ctypes.CDLL(str(LIB))
        L = self.lib
        L.pe_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64,
                              ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int]
        L.pe_open.restype = ctypes.c_void_p
        L.pe_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32,
                             ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_PeTiming)]
        L.pe_run.restype = ctypes.c_int64
        L.pe_run_unpacked.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32,
                                      ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                                      ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_PeTiming)]
        L.pe_run_unpacked.restype = ctypes.c_int64
        L.pe_last_error.argtypes = [ctypes.c_void_p]
        L.pe_last_error.restype = ctypes.c_char_p
        L.pe_info.argtypes = [ctypes.c_void_p]
        L.pe_info.restype = ctypes.c_char_p
        L.pe_close.argtypes = [ctypes.c_void_p]
        self.h = L.pe_open(str(xclbin).encode(), str(device).encode(), d, max_hits, max_events, max_cand, kernel.encode(),
                           int(mode))
        if not self.h:
            raise RuntimeError(f"pe_open failed: {L.pe_last_error(None).decode()}")
        self.d, self.max_cand, self.timeout_ms = d, max_cand, timeout_ms
        self.info = json.loads(L.pe_info(self.h).decode())
        self.last_timing = {}
        self._out = np.zeros(max_cand, np.uint64)
        self._t = _PeTiming()
        # result buffers of run_unpacked (reused: the returned arrays are views, valid until the next call)
        self._rows = np.zeros(max_cand, np.int32)
        self._l0 = np.zeros(max_cand, np.float32)
        self._l1 = np.zeros(max_cand, np.float32)
        self._cnt = np.zeros(max_events, np.int32)
        self._nh = np.zeros(max_events, np.int32)
        self._ptrs = (ctypes.c_void_p * max_events)()
        self._addr = [a.ctypes.data for a in (self._rows, self._l0, self._l1, self._cnt, self._nh)]
        self._tref = ctypes.byref(self._t)
        self.max_events = max_events

    @classmethod
    def from_meta(cls, meta_path, device="0", **kw):
        meta = json.loads(Path(meta_path).read_text())
        xclbin = Path(meta_path).parent / meta["xclbin"]
        key = (str(xclbin), kw.get("mode", 1))
        if key not in cls._instances:
            cls._instances[key] = cls(xclbin, d=meta["d"], device=device, kernel=meta.get("kernel", "pairedges"), **kw)
        inst = cls._instances[key]
        inst.meta = meta
        return inst

    def run(self, embs, thr_raw):
        import hi_common as H
        n, x, cap = _pack_embeddings(embs, self.d)
        cnt = np.zeros(len(embs), np.int32)
        used = self.lib.pe_run(self.h, x.ctypes.data, n.ctypes.data, len(embs), int(thr_raw), self._out.ctypes.data,
                               self.max_cand, cnt.ctypes.data, self.timeout_ms, ctypes.byref(self._t))
        if used < 0:
            raise RuntimeError(f"pe_run failed: {self.lib.pe_last_error(self.h).decode()}")
        t = self._t
        self.last_timing = {k: getattr(t, k) for k in ("copy_in_us", "h2d_us", "kernel_us", "d2h_us", "copy_out_us", "total_us")}
        return H.split_candidates(self._out[:used].copy(), cnt), dict(self.last_timing)

    def run_unpacked(self, embs, thr_raw, frac):
        """Candidates unpacked by the driver: (rows int32, l0 float32, l1 float32) over all events in order,
        and the count per event. Each emb must be a C-contiguous float32 [N, d] array (as ORT returns them).
        The returned arrays are views of reused buffers, valid until the next call."""
        n = len(embs)
        if n > self.max_events:
            raise ValueError(f"more than {self.max_events} events per call")
        for k, e in enumerate(embs):
            if e.dtype != np.float32 or not e.flags.c_contiguous or e.ndim != 2 or e.shape[1] != self.d:
                e = np.ascontiguousarray(e, np.float32).reshape(-1, self.d)
                embs[k] = e
            self._ptrs[k] = e.ctypes.data
            self._nh[k] = e.shape[0]
        ar, a0, a1, ac, an = self._addr
        used = self.lib.pe_run_unpacked(self.h, self._ptrs, an, n, int(thr_raw), int(frac), ar, a0, a1, self.max_cand, ac,
                                        self.timeout_ms, self._tref)
        if used < 0:
            raise RuntimeError(f"pe_run_unpacked failed: {self.lib.pe_last_error(self.h).decode()}")
        return self._rows[:used], self._l0[:used], self._l1[:used], self._cnt[:n]

    @property
    def timing(self):
        t = self._t
        return {k: getattr(t, k) for k in ("copy_in_us", "h2d_us", "kernel_us", "d2h_us", "copy_out_us", "total_us")}

    def close(self):
        if self.h:
            self.lib.pe_close(self.h)
            self.h = None

