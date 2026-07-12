"""MTP export stage A (m3vllm python): HF bf16 mtp.* -> f32 npy + manifest.

Splits experts.gate_up_proj into switch_mlp gate/up (row midpoint, axis 1),
pre-shifts all 7 MTP norm tensors by +1.0 (MLX RMSNorm convention — the oMLX
sanitize per-key heuristic (shift iff mean<0.5) would misjudge raw Qwen3.6
mtp.norm at HF mean 1.93), and emits a manifest stage B consumes.

env: NPY=/var/hf/qwen36_mtp_npy
"""
import json
import os

import numpy as np
import torch
from safetensors import safe_open

MODELDIR = "/var/hf/Qwen3.6-35B-A3B"
OUT = os.environ.get("NPY", "/var/hf/qwen36_mtp_npy")
os.makedirs(OUT, exist_ok=True)
wm = json.load(open(os.path.join(MODELDIR, "model.safetensors.index.json")))["weight_map"]
names = sorted(k for k in wm if k.startswith("mtp."))
assert names, "no mtp tensors in checkpoint"

NORM_SUFFIXES = ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "q_norm.weight", "k_norm.weight", "mtp.norm.weight",
                 "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight")
QSPEC = {  # module suffix -> (bits, group_size); absent = raw fp
    "mlp.switch_mlp.gate_proj": (4, 64),
    "mlp.switch_mlp.up_proj": (4, 64),
    "mlp.switch_mlp.down_proj": (4, 64),
    "mlp.shared_expert.gate_proj": (8, 128),
    "mlp.shared_expert.up_proj": (8, 128),
    "mlp.shared_expert.down_proj": (8, 128),
    "mlp.shared_expert_gate": (8, 64),
    "self_attn.q_proj": (8, 64),
    "self_attn.k_proj": (8, 64),
    "self_attn.v_proj": (8, 64),
    "self_attn.o_proj": (8, 64),
}

manifest = []  # {kind: quant|raw, name (module or tensor), file, bits, gs}


def emit(art_name, arr):
    """art_name = full artifact tensor name (with .weight for linears)."""
    fn = art_name.replace("/", "_") + ".npy"
    np.save(os.path.join(OUT, fn), arr.numpy())
    module = art_name[:-len(".weight")] if art_name.endswith(".weight") else art_name
    spec = next((q for sfx, q in QSPEC.items() if module.endswith(sfx)), None)
    if spec is not None:
        manifest.append({"kind": "quant", "name": module, "file": fn,
                         "bits": spec[0], "gs": spec[1]})
    else:
        manifest.append({"kind": "raw", "name": art_name, "file": fn})
    print(f"{art_name:70s} {tuple(arr.shape)} {'q' + str(spec) if spec else 'raw'}")


for k in names:
    with safe_open(os.path.join(MODELDIR, wm[k]), framework="pt") as f:
        t = f.get_tensor(k).float()
    if k.endswith(NORM_SUFFIXES):
        # Ship final MLX-convention values (w+1). The Mac loader shields these
        # from the oMLX sanitize magnitude heuristic (which double-shifts any
        # shipped mean < 0.5 — pre_fc_norm_embedding lands at 0.27).
        t = t + 1.0
        print(f"  norm {k}: shipped mean {float(t.mean()):.3f}")
    art = "language_model." + k
    if k == "mtp.layers.0.mlp.experts.gate_up_proj":
        half = t.shape[1] // 2
        base = "language_model.mtp.layers.0.mlp.switch_mlp"
        emit(f"{base}.gate_proj.weight", t[:, :half].contiguous())
        emit(f"{base}.up_proj.weight", t[:, half:].contiguous())
    elif k == "mtp.layers.0.mlp.experts.down_proj":
        emit("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight", t.contiguous())
    else:
        emit(art, t.contiguous())

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"saved {len(manifest)} entries -> {OUT}")
print("MTP_STAGEA_DONE")
