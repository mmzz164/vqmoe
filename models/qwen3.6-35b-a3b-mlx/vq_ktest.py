"""Synthetic kernel correctness + bench: all (d, nbits) tiers vs level-0 reference.

Covers the 2.0bpw mixed menu: tier2 (d4/K256/8bit), tier3 (d4/K4096/12bit),
tier1.5 (d8/K4096/12bit) on both expert shapes (gate/up R512xC2048, down R2048xC512),
plus the fused swiglu kernels (d4, d8). No model artifact needed.
"""
import os
import time

import numpy as np
import mlx.core as mx

import vq_kernel
import vq_switch


def pack_np(codes, nbits):
    """codes [rows, nsub] -> LSB-first packed uint32 [rows, PC] (mirrors CUDA pack_codes)."""
    rows, nsub = codes.shape
    PC = (nsub * nbits + 31) // 32
    out = np.zeros((rows, PC), dtype=np.uint64)
    for k in range(nsub):
        pos = k * nbits
        w, off = divmod(pos, 32)
        c = codes[:, k].astype(np.uint64)
        out[:, w] |= (c << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        if off + nbits > 32:
            out[:, w + 1] |= c >> np.uint64(32 - off)
    return out.astype(np.uint32)


def build_module(E, R, C, d, K, nbits, seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, K, size=(E * R, C // d))
    packed = pack_np(codes, nbits).reshape(E, R, -1)
    scales = (rng.random((E, R, C // 128), dtype=np.float32) * 0.5 + 0.25).astype(np.float16)
    cb = (rng.standard_normal((K, d)) * 0.05).astype(np.float16)
    m = vq_switch.VQSwitchLinear(E, R, C, nbits, d)
    m.vq_codes = mx.array(packed.view(np.int32))
    m.vq_scales = mx.array(scales)
    os.environ["VQ_KERNEL"] = "0"          # level-0 reference path
    m.set_codebook(mx.array(cb))
    return m


def rel_err(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.abs(a32 - b32).max()) / max(float(mx.abs(a32).max()), 1e-9)


CASES = [  # (tier label, d, K, nbits, gemv fn)
    ("t2  d4 K256  8b ", 4, 256, 8, vq_kernel.vq_gemv2),
    ("t3  d4 K4096 12b", 4, 4096, 12, vq_kernel.vq_gemv2),
    ("t15 d8 K4096 12b", 8, 4096, 12, vq_kernel.vq_gemv2_d8),
]
SHAPES = [("gateup", 16, 512, 2048), ("down  ", 16, 2048, 512)]
N = 8
fails = 0

print("== GEMV kernel vs level-0 ==")
for label, d, K, nbits, gemv in CASES:
    for sname, E, R, C in SHAPES:
        m = build_module(E, R, C, d, K, nbits, seed=hash((d, K, C)) % 2**31)
        rng = np.random.default_rng(99)
        x = mx.array((rng.standard_normal((N, 1, C)) * 0.3).astype(np.float16))
        eidx = mx.array(rng.integers(0, E, size=(N,)).astype(np.int32))
        y_ref = m(x, eidx)
        mx.eval(y_ref)
        y_k = gemv(x.reshape(N, C), m._codes_u32, m._sc16, m._cb, eidx, nbits)
        mx.eval(y_k)
        e = rel_err(y_ref.reshape(N, -1), y_k)
        ok = e < 2e-2
        fails += 0 if ok else 1
        print(f"  {label} {sname}: rel={e:.3e} {'OK' if ok else 'FAIL'}")

print("== fused swiglu vs level-0 ==")
for label, d, K, nbits, _ in CASES:
    swiglu = {4: vq_kernel.vq_swiglu, 8: vq_kernel.vq_swiglu_d8}[d]
    E, R, C = 16, 512, 2048
    mg = build_module(E, R, C, d, K, nbits, seed=1234)
    mu = build_module(E, R, C, d, K, nbits, seed=5678)
    mu.set_codebook(mg._cb)                 # fused kernel shares one codebook
    rng = np.random.default_rng(7)
    x = mx.array((rng.standard_normal((N, 1, C)) * 0.3).astype(np.float16))
    eidx = mx.array(rng.integers(0, E, size=(N,)).astype(np.int32))
    yg = mg(x, eidx).astype(mx.float32)
    yu = mu(x, eidx).astype(mx.float32)
    inter_ref = (mx.sigmoid(yg) * yg * yu).reshape(N, -1)
    mx.eval(inter_ref)
    y_k = swiglu(x.reshape(N, C), mg._codes_u32, mg._sc16, mu._codes_u32, mu._sc16,
                 mg._cb, eidx, nbits)
    mx.eval(y_k)
    e = rel_err(inter_ref, y_k)
    ok = e < 2e-2
    fails += 0 if ok else 1
    print(f"  {label} swiglu: rel={e:.3e} {'OK' if ok else 'FAIL'}")

print("== prefill fast path (dequant+gather_mm) vs level-0, N=512 ==")
for label, d, K, nbits, _ in CASES:
    E, R, C = 16, 512, 2048
    m = build_module(E, R, C, d, K, nbits, seed=99)
    rng = np.random.default_rng(3)
    N = 512
    x = mx.array((rng.standard_normal((N, 1, C)) * 0.3).astype(np.float16))
    eidx = mx.array(np.sort(rng.integers(0, E, size=(N,))).astype(np.int32))
    os.environ["VQ_PREFILL_N"] = "256"
    y_fast = m(x, eidx, sorted_indices=True)
    mx.eval(y_fast)
    os.environ["VQ_PREFILL_N"] = "0"
    y_ref = m(x, eidx, sorted_indices=True)
    mx.eval(y_ref)
    e = rel_err(y_ref, y_fast)
    ok = e < 2e-2
    fails += 0 if ok else 1
    print(f"  {label} gather_mm: rel={e:.3e} {'OK' if ok else 'FAIL'}")

print("== GLU prefill fast path vs fallback, 64 tok x8 ==")
for label, d, K, nbits, _ in CASES:
    mg = build_module(16, 512, 2048, d, K, nbits, seed=11)
    mu = build_module(16, 512, 2048, d, K, nbits, seed=22)
    md = build_module(16, 2048, 512, d, K, nbits, seed=33)
    mu.set_codebook(mg._cb)
    md.set_codebook(md._cb)                     # own book (down may differ in prod)
    glu = vq_switch.VQSwitchGLU(mg, mu, md)
    rng = np.random.default_rng(5)
    x = mx.array((rng.standard_normal((64, 2048)) * 0.3).astype(np.float16))
    idx = mx.array(rng.integers(0, 16, size=(64, 8)).astype(np.int32))
    os.environ["VQ_PREFILL_N"] = "256"
    y_fast = glu(x, idx)
    mx.eval(y_fast)
    os.environ["VQ_PREFILL_N"] = "0"
    y_ref = glu(x, idx)                          # kernels are None -> level-0 flow
    mx.eval(y_ref)
    e = rel_err(y_ref, y_fast)
    ok = e < 2e-2
    fails += 0 if ok else 1
    print(f"  {label} glu_fast: rel={e:.3e} {'OK' if ok else 'FAIL'}")
os.environ["VQ_PREFILL_N"] = "256"

print("== bench (N=8 tokens, us/dispatch, median of 50) ==")
for label, d, K, nbits, gemv in CASES:
    for sname, E, R, C in [("gateup", 64, 512, 2048), ("down  ", 64, 2048, 512)]:
        m = build_module(E, R, C, d, K, nbits, seed=42)
        rng = np.random.default_rng(1)
        x = mx.array((rng.standard_normal((N, C)) * 0.3).astype(np.float16))
        eidx = mx.array(rng.integers(0, E, size=(N,)).astype(np.int32))
        for _ in range(5):
            mx.eval(gemv(x, m._codes_u32, m._sc16, m._cb, eidx, nbits))
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            mx.eval(gemv(x, m._codes_u32, m._sc16, m._cb, eidx, nbits))
            ts.append((time.perf_counter() - t0) * 1e6)
        print(f"  {label} {sname}: {sorted(ts)[len(ts)//2]:.0f} us")

print("KTEST_FAILS", fails)
print("VQ_KTEST_DONE" if fails == 0 else "VQ_KTEST_FAILED")
