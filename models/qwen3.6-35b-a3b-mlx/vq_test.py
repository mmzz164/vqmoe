"""Mac-side VQ shim test (module level): golden check + micro-benchmark.

Run on the Mac:  python vq_test.py <artifact_dir> <golden.npz>
1. Metal smoke: trivial mx.fast.metal_kernel JIT + run (proves SSH+Metal+no-keychain).
2. Golden: build VQSwitchLinear for the two reference modules straight from artifact
   tensors, decode, y = x @ W_e.T, compare vs CUDA golden (fp16 tolerance).
3. Micro-bench: decode+matmul timing for a [1 tok, 8 experts] call (level-0 speed).
"""
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vq_switch import VQSwitchLinear

ART = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.6-35B-A3B-MLX-VQ-map12")
GOLD = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/vq_golden.npz")

print(f"mlx {mx.__version__} device={mx.default_device()}")

# ---- 1. Metal custom-kernel smoke ----
src = """
    uint i = thread_position_in_grid.x;
    if (i < inp_shape[0]) { out[i] = inp[i] * 2.0f + 1.0f; }
"""
try:
    k = mx.fast.metal_kernel(name="smoke", input_names=["inp"], output_names=["out"], source=src)
    a = mx.arange(8, dtype=mx.float32)
    (o,) = k(inputs=[a], output_shapes=[a.shape], output_dtypes=[mx.float32],
             grid=(8, 1, 1), threadgroup=(8, 1, 1))
    mx.eval(o)
    ok = bool(mx.all(o == a * 2 + 1))
    print(f"[1] metal_kernel JIT smoke: {'PASS' if ok else 'FAIL'} ({o.tolist()[:4]}...)")
except Exception as e:
    print(f"[1] metal_kernel smoke FAILED: {e!r}")

# ---- 2. golden check ----
cfg = json.load(open(os.path.join(ART, "config.json")))
meta = cfg["vq"]["modules"]
idx = json.load(open(os.path.join(ART, "model.safetensors.index.json")))["weight_map"]
cbs = mx.load(os.path.join(ART, cfg["vq"]["codebooks_file"]))
g = np.load(GOLD)
mods = sorted(set(k.rsplit("__", 1)[0].replace("__", ".") for k in g.files))
worst = 0.0
for m in mods:
    info = meta[m]
    key = m.replace(".", "__")
    x = mx.array(g[key + "__x"])                       # [4, C]
    y_ref = g[key + "__y"]                             # [4exp, 4tok, R]
    experts = [int(e) for e in g[key + "__experts"]]
    shard = mx.load(os.path.join(ART, idx[m + ".vq_codes"]))
    codes = shard[m + ".vq_codes"]
    scales = mx.load(os.path.join(ART, idx[m + ".vq_scales"]))[m + ".vq_scales"]
    R = codes.shape[1]
    vq = VQSwitchLinear(num_experts=codes.shape[0], output_dims=R,
                        input_dims=info["in_dims"], nbits=info["nbits"], d=info["d"],
                        norm_group=info["norm_group"])
    vq.vq_codes = codes
    vq.vq_scales = scales
    vq.set_codebook(cbs[f"cb{info['vq_bits']}"])
    W = vq._decode(mx.array(experts))                  # [4exp, R, C]
    y = mx.matmul(x[None].astype(W.dtype), mx.swapaxes(W, -1, -2))   # [4exp, 4tok, R]
    mx.eval(y)
    err = float(mx.abs(y - mx.array(y_ref)).max())
    rel = err / (abs(y_ref).max() + 1e-9)
    worst = max(worst, rel)
    print(f"[2] {m.split('layers.')[1]}: max|dy|={err:.4e} rel={rel:.2e} {'PASS' if rel < 2e-2 else 'FAIL'}")
print(f"[2] golden worst rel = {worst:.2e} -> {'PASS' if worst < 2e-2 else 'FAIL'}")

# ---- 3. kernel vs level-0: correctness then speed ----
os.environ["VQ_KERNEL"] = "1"
x1 = mx.random.normal((1, 1, 1, vq._C)).astype(mx.float16)
idx8 = mx.array([[0, 5, 17, 42, 99, 123, 200, 255]])
vq.set_codebook(vq._cb)                                  # rebind with kernel enabled
yk = vq(x1, idx8); mx.eval(yk)
vq._gemv = None                                          # force level-0
y0 = vq(x1, idx8).astype(mx.float16); mx.eval(y0)
kerr = float(mx.abs(yk.astype(mx.float32) - y0.astype(mx.float32)).max())
kref = float(mx.abs(y0).max())
print(f"[3] kernel-vs-level0: max|dy|={kerr:.4e} rel={kerr/(kref+1e-9):.2e} "
      f"{'PASS' if kerr/(kref+1e-9) < 2e-2 else 'FAIL'}")

def bench(fn, n=50):
    for _ in range(5): mx.eval(fn())
    t = time.time()
    for _ in range(n): mx.eval(fn())
    return (time.time() - t) / n * 1000

vq.set_codebook(vq._cb)                                  # kernel on
tk = bench(lambda: vq(x1, idx8))
vq._gemv = None
t0b = bench(lambda: vq(x1, idx8), n=20)
print(f"[4] 1tok x 8exp: kernel {tk:.3f} ms vs level-0 {t0b:.2f} ms  (x{t0b/max(tk,1e-6):.1f} speedup)"
      f"  -> layer-stack est ≈ {tk*120:.1f} ms/token ≈ {1000/max(tk*120,1e-6):.1f} tok/s bound")
print("VQ_TEST_DONE")
