# SPDX-License-Identifier: Apache-2.0
"""AutoRound(auto_round:auto_gptq, mixed 2/3/4/8-bit, v1) -> vLLM quant config.

Reuses OneComp's MixedGPTQConfig + its numerically-validated v1 kernels UNCHANGED
(grouped_moe / fused_dq_gemm / mixed_moe). The ONLY adaptation is parsing: AutoRound
stores per-module bits in a flat `extra_config` keyed by full module path + a global
`bits`(=16 "unquantized unless overridden"), `group_size`, `sym`; we convert that to
the plugin's per-layer `quantization_bits` list-of-dicts {suffix: {bits, method}}.
Non-overridden (global-16) modules are omitted -> dispatch returns UnquantizedLinear
(BF16) for router gate / indexer / norms / embed / lm_head.

Verified (see VLLM_M3_PORT_SPEC.md): all non-expert quantized modules are 4/8-bit
(stock AutoGPTQ Marlin), only routed experts are mixed 2/3/4/8 (MixedGPTQMoEMethod).
v1 +1 dequant convention is identical between mixed_gptq and auto_round:auto_gptq, so
the kernels are reused with zero numeric change -> no quantization-side degradation.
"""
import re
import sys

sys.path.insert(0, "/home/mizugaihiros01/work/onecomp")

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm_plugins.gptq.vllm_plugin import MixedGPTQConfig

# Our AutoRound checkpoint stores gate_up_proj ALREADY-FUSED (one quantized tensor),
# unlike mixed_gptq which had separate gate_proj/up_proj. Drop it from the plugin's
# fused-constituent table so the within-shard validator treats it as a single module
# (qkv stays fused: the checkpoint DOES store q/k/v separately). Plugin file unchanged.
from vllm_plugins.utils import module as _oc_module
_oc_module._FUSED_TO_CONSTITUENTS.pop("gate_up_proj", None)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.(.+)$")


def autoround_extra_config_to_quantization_bits(extra_config: dict) -> list:
    """Flat AutoRound extra_config -> per-layer list[ {suffix: {bits, method}} ]."""
    num_layers = 0
    parsed = []
    for full_key, v in extra_config.items():
        bits = v.get("bits") if isinstance(v, dict) else None
        if bits is None or bits >= 16:
            continue  # global-16 / unquantized -> omit -> BF16
        m = _LAYER_RE.search(full_key)
        if not m:
            continue
        layer_idx, suffix = int(m.group(1)), m.group(2)
        num_layers = max(num_layers, layer_idx + 1)
        parsed.append((layer_idx, suffix, int(bits)))
    qbits = [dict() for _ in range(num_layers)]
    for layer_idx, suffix, bits in parsed:
        cfg = {"bits": bits, "method": "gptq"}
        qbits[layer_idx][suffix] = cfg
        # The official vLLM M3 renames the shared expert mlp.shared_experts.* ->
        # block_sparse_moe.shared_experts.*, and _lookup_module_config matches by
        # `official_suffix.startswith(cfg_name)`. Without an alias the official
        # prefix misses the ckpt-named bits -> the module falls to bf16
        # (UnquantizedLinearMethod) -> wasted ~0.85 GiB/rank for down_proj (and
        # it then fails to load our quantized weight). Alias keeps it quantized.
        # (gate_up_proj is still forced bf16 by get_quant_method's override since
        # our fused gate_up can't load as one GPTQ tensor; this only helps the
        # single-tensor down_proj.)
        if suffix.startswith("mlp.shared_experts."):
            alias = "block_sparse_moe.shared_experts." + suffix[len("mlp.shared_experts."):]
            qbits[layer_idx][alias] = cfg
    return qbits


@register_quantization_config("autoround_mixed")
class AutoRoundMixedConfig(MixedGPTQConfig):
    """MixedGPTQConfig fed from an AutoRound quantization_config dict."""

    @classmethod
    def get_name(cls) -> str:
        return "autoround_mixed"

    # The official vLLM M3 model FUSES gate_up (MergedColumnParallelLinear) and
    # q/k/v(+indexer) (QKVParallelLinear / MinimaxM3QKVParallelLinearWithIndexer).
    # Our checkpoint stores gate_up pre-fused and the sparse-layer indexer q/k in
    # bf16, so these two fused linears can't be loaded as a single GPTQ tensor.
    # m3_official_loader de-quants them to bf16; mark them unquantized here so the
    # model builds bf16 linears that accept those weights. Everything else
    # (routed experts, o_proj, down_proj, dense attn) stays on the normal path.
    #
    # GATED behind M3_OFFICIAL_PORT=1 (set only by serve_m3_official.py): the M1
    # production model (m3_api_server.py) ALSO uses autoround_mixed and KEEPS
    # gate_up/qkv quantized, so this override must NOT apply there (else KeyError
    # 'qweight' at forward — the linear has no qweight after being unquantized).
    def get_quant_method(self, layer, prefix: str):
        import os
        # Only the sparse-attn fused qkv_proj MUST be bf16 (q/k/v are quantized
        # but fused with the bf16 indexer q/k -> can't be one GPTQ tensor;
        # m3_official_loader de-quants it). gate_up_proj USED to be forced bf16
        # too, but it is now kept QUANTIZED: the loader splits the fused GPTQ
        # gate_up into separate quantized gate_proj/up_proj shards (no de-quant)
        # -> reclaims ~2 GiB/rank for KV. (bits found via the per-module config;
        # the shared expert relies on the block_sparse_moe.* alias added in
        # autoround_extra_config_to_quantization_bits.)
        if (os.environ.get("M3_OFFICIAL_PORT") == "1"
                and isinstance(layer, LinearBase)
                and prefix.endswith(".self_attn.qkv_proj")):
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)

    @classmethod
    def from_config(cls, config: dict) -> "AutoRoundMixedConfig":
        extra = config.get("extra_config", {}) or {}
        qbits = autoround_extra_config_to_quantization_bits(extra)
        return cls(
            quantization_bits=qbits,
            group_size=config.get("group_size", 128),
            desc_act=False,
            sym=config.get("sym", True),
            lm_head_quantized=False,
            checkpoint_format="gptq",   # auto_round:auto_gptq == GPTQ v1
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    cfgp = "/var/hf/MiniMax-M3-AutoRound-3.2bit-i1000/MiniMax-M3-w16g128/quantization_config.json"
    cfg = json.load(open(cfgp))
    qb = autoround_extra_config_to_quantization_bits(cfg.get("extra_config", {}))
    print(f"layers in quantization_bits: {len(qb)}")
    # spot-check a dense, a moe attn, and an expert
    def show(L, needle):
        hit = {k: v for k, v in qb[L].items() if needle in k}
        # compress: print count + a sample
        sample = dict(list(hit.items())[:2])
        print(f"  L{L} [{needle}]: n={len(hit)} sample={sample}")
    show(0, "self_attn.q_proj")
    show(0, "mlp.gate_up_proj")
    show(3, "self_attn.q_proj")
    show(3, "self_attn.o_proj")
    show(3, "mlp.experts.0.")
    show(3, "mlp.shared_experts")
    show(59, "self_attn.o_proj")
    # sanity: total quantized modules
    tot = sum(len(d) for d in qb)
    from collections import Counter
    bitc = Counter(v["bits"] for d in qb for v in d.values())
    print(f"total quantized modules: {tot} | bits histogram: {dict(sorted(bitc.items()))}")
    print("M3_QUANT_SELFTEST_OK")
