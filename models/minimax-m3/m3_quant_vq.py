# SPDX-License-Identifier: Apache-2.0
"""VQ-aware variant of m3_quant: registers "autoround_mixed_vq".

The stock parser (m3_quant.autoround_extra_config_to_quantization_bits) reduces
every extra_config entry to {"bits", "method"} — dropping the {"format": "vq",
"d": ...} marker that the M3-VQ build (m3_vq_build.py) writes on routed experts.
Without the marker, MixedGPTQConfig.get_quant_method dispatches the experts to
the SCALAR MixedGPTQMoEMethod, whose buffers don't match VQ codes -> load-time
shape mismatch. This module is ADDITIVE: m3_quant.py is untouched (the M1 and
② scalar production paths are unaffected); the VQ server (m3_vq_api_server.py)
selects quantization="autoround_mixed_vq" explicitly.

With the marker preserved, the inherited dispatch (vllm_plugin.py) routes the
experts to MixedVQMoEMethod (vq_moe.py) — the GLM-5.2-proven VQ serving method,
which is shape-agnostic (derives hid/inter/gs from the layer) and loads the
shared codebooks from VQ_CODEBOOKS_DIR (default /var/hf/glm_quant; transfer test
2026-07-10: GLM codebooks are M3-optimal within noise). Everything else
(qkv/gate_up handling, spine Marlin, key translation) is inherited unchanged.
"""
import sys

sys.path.insert(0, "/var/hf/vllm_m3")

import m3_quant  # noqa: F401  (base registration + _FUSED_TO_CONSTITUENTS patch + onecomp path)
from m3_quant import (
    _LAYER_RE,
    AutoRoundMixedConfig,
    autoround_extra_config_to_quantization_bits,
)
from vllm.model_executor.layers.quantization import register_quantization_config


def autoround_extra_config_to_quantization_bits_vq(extra_config: dict) -> list:
    """Stock parser + a second pass that carries the VQ marker into the cfgs."""
    qbits = autoround_extra_config_to_quantization_bits(extra_config)
    carried = 0
    for full_key, v in extra_config.items():
        if not (isinstance(v, dict) and v.get("format") == "vq"):
            continue
        m = _LAYER_RE.search(full_key)
        if not m:
            continue
        layer_idx, suffix = int(m.group(1)), m.group(2)
        cfg = qbits[layer_idx].get(suffix) if layer_idx < len(qbits) else None
        if cfg is not None:
            cfg["format"] = "vq"
            if "d" in v:
                cfg["d"] = v["d"]
            carried += 1
    print(f"[m3_quant_vq] carried format:'vq' through for {carried} expert modules", flush=True)
    return qbits


@register_quantization_config("autoround_mixed_vq")
class AutoRoundMixedVQConfig(AutoRoundMixedConfig):
    """AutoRoundMixedConfig whose parser preserves the per-expert VQ marker."""

    @classmethod
    def get_name(cls) -> str:
        return "autoround_mixed_vq"

    @classmethod
    def from_config(cls, config: dict) -> "AutoRoundMixedVQConfig":
        extra = config.get("extra_config", {}) or {}
        qbits = autoround_extra_config_to_quantization_bits_vq(extra)
        return cls(
            quantization_bits=qbits,
            group_size=config.get("group_size", 128),
            desc_act=False,
            sym=config.get("sym", True),
            lm_head_quantized=False,
            checkpoint_format="gptq",
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    cfgp = "/var/hf/MiniMax-M3-VQ-r1/MiniMax-M3-w16g128/quantization_config.json"
    cfg = json.load(open(cfgp))
    qb = autoround_extra_config_to_quantization_bits_vq(cfg.get("extra_config", {}))
    from collections import Counter
    vq = Counter()
    for d in qb:
        for k, v in d.items():
            if v.get("format") == "vq":
                vq[(v["bits"], v["d"])] += 1
    tot = sum(len(d) for d in qb)
    print(f"layers: {len(qb)}  total modules: {tot}")
    print(f"vq (bits,d) histogram: {dict(sorted(vq.items()))}")
    spine = sum(1 for d in qb for v in d.values() if "format" not in v)
    print(f"scalar spine modules: {spine}")
    assert sum(vq.values()) == 21888, sum(vq.values())
    print("M3_QUANT_VQ_SELFTEST_OK")
