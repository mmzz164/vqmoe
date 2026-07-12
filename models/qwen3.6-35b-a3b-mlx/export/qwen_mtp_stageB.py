"""MTP export stage B (mlxenv python): quantize per contract, append to artifact.

Reads stage A npys + manifest, mx.quantize's the quantized modules, writes
model-mtp.safetensors into the target artifact, updates the safetensors index
(total_size included) and the config quantization overrides (every quantized
mtp module listed EXPLICITLY, incl. 4/64 defaults — the Mac loader predicate
quantizes an mtp module iff it appears in the config quantization dict).

Run: LD_LIBRARY_PATH=$MLXENV/lib/python3.13/site-packages/mlx/lib $MLXENV/bin/python
env: NPY=/var/hf/qwen36_mtp_npy ART=/var/hf/Qwen3.6-35B-A3B-MLX-VQ-2.4bpw
"""
import json
import os

import numpy as np
import mlx.core as mx

NPY = os.environ.get("NPY", "/var/hf/qwen36_mtp_npy")
ART = os.environ.get("ART", "/var/hf/Qwen3.6-35B-A3B-MLX-VQ-2.4bpw")
manifest = json.load(open(os.path.join(NPY, "manifest.json")))

idx_path = os.path.join(ART, "model.safetensors.index.json")
idx = json.load(open(idx_path))
wm = idx["weight_map"]

# match artifact dtypes: scales/biases like backbone quantized modules, raw like norms
probe_q = "language_model.model.layers.0.mlp.shared_expert.gate_proj.scales"
SB_DTYPE = mx.load(os.path.join(ART, wm[probe_q]))[probe_q].dtype
probe_fp = "language_model.model.norm.weight"
FP_DTYPE = mx.load(os.path.join(ART, wm[probe_fp]))[probe_fp].dtype
print(f"scales dtype={SB_DTYPE} fp dtype={FP_DTYPE}")

tensors, overrides = {}, {}
for e in manifest:
    W = mx.array(np.load(os.path.join(NPY, e["file"])))
    if e["kind"] == "quant":
        wq, sc, bi = mx.quantize(W, group_size=e["gs"], bits=e["bits"])
        deq = mx.dequantize(wq, sc, bi, group_size=e["gs"], bits=e["bits"])
        rel = float(mx.mean((deq - W) ** 2) / mx.maximum(mx.mean(W ** 2), 1e-12))
        tensors[e["name"] + ".weight"] = wq
        tensors[e["name"] + ".scales"] = sc.astype(SB_DTYPE)
        tensors[e["name"] + ".biases"] = bi.astype(SB_DTYPE)
        overrides[e["name"]] = {"group_size": e["gs"], "bits": e["bits"], "mode": "affine"}
        print(f"q{e['bits']}/gs{e['gs']} {e['name']}: rel_mse={rel:.2e} wq{tuple(wq.shape)}")
    else:
        tensors[e["name"]] = W.astype(FP_DTYPE)
        print(f"raw      {e['name']}: {tuple(W.shape)} mean={float(mx.mean(W)):.3f}")

shard = "model-mtp.safetensors"
mx.save_safetensors(os.path.join(ART, shard), tensors, metadata={"format": "mlx"})

added = 0
for name, arr in tensors.items():
    assert name not in wm, f"{name} already in index"
    wm[name] = shard
    added += arr.nbytes
idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + added
json.dump(idx, open(idx_path, "w"), indent=0)

cfg_path = os.path.join(ART, "config.json")
cfg = json.load(open(cfg_path))
for qk in ("quantization", "quantization_config"):
    q = cfg.get(qk) or {}
    q.update(overrides)
    cfg[qk] = q
json.dump(cfg, open(cfg_path, "w"), indent=1)

print(f"appended {len(tensors)} tensors ({added/1e9:.3f} GB) -> {shard}; "
      f"{len(overrides)} quant overrides")
print("MTP_STAGEB_DONE")
