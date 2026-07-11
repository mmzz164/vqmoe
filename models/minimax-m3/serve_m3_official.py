# SPDX-License-Identifier: Apache-2.0
"""Load our AutoRound 3.2bit (mixed_gptq) checkpoint on the OFFICIAL vLLM M3
model (vLLM 0.23.1 nightly, native MiniMax-M3 + MSA sparse-attention indexer).

Unlike serve_test.py (our out-of-tree M1 model), here vLLM's NATIVE M3 model is
used (registered by the wheel). We only supply the quantization: register
"autoround_mixed" (our mixed_gptq plugin) + the INC bypass. block-size 128 is
mandatory for MSA. Long context comes from the native indexer for free.

Run in the m3vllm env:
  /home/mizugaihiros01/anaconda3/envs/m3vllm/bin/python /var/hf/vllm_m3/serve_m3_official.py
"""
import os, sys, time
os.environ["M3_OFFICIAL_PORT"] = "1"   # activates m3_quant's gate_up/qkv unquant
                                       # override (M1 leaves it unset -> quantized)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MIXED_MOE_GROUPED", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, "/var/hf/vllm_m3")

CKPT = "/var/hf/MiniMax-M3-AutoRound-3.2bit-i1000/MiniMax-M3-w16g128"
MAXLEN = int(os.environ.get("M3_MAXLEN", "8192"))   # exercise the indexer (>2048)
t0 = time.time()

import m3_quant   # noqa: registers "autoround_mixed" + _FUSED_TO_CONSTITUENTS patch
                 # (runs on every (re)import, incl. spawn workers -> propagates)
import m3_official_loader  # noqa: registers the key-translating CausalLM loader
from vllm import LLM, SamplingParams


def _override_quant_method(config):
    qc = getattr(config, "quantization_config", None)
    if isinstance(qc, dict) and qc.get("quant_method") == "auto-round":
        qc["quant_method"] = "autoround_mixed"
    # Force the TEXT-ONLY backbone: our long-context goal is text, the official
    # VL wrapper would also require loading the vision tower (different naming),
    # and m3_official_loader registers a subclass for exactly this arch. The
    # vision-capable path stays on the out-of-tree M1 model.
    config.architectures = ["MiniMaxM3SparseForCausalLM"]
    # CRITICAL: the engine loads the checkpoint's OWN config class (proven: the
    # first attempt saw hidden_act="silu", not vLLM's swigluoai default), and the
    # official model reads several values as TOP-LEVEL attributes that our
    # checkpoint config stores elsewhere or omits:
    #   - config.rope_theta  (ours nests it in rope_parameters -> top-level MISSING
    #     -> wrong RoPE theta -> garbage; M3 uses theta=5e6)
    #   - config.swiglu_beta (omitted -> M3 uses beta=1.0: (up+1)*gate*sig(1.702*gate))
    #   - config.hidden_act  ("silu" -> must be "swigluoai")
    # Force the known-correct MiniMax-M3 values (match vLLM's MiniMaxM3Config
    # defaults; these are the values the M1 model is validated against).
    FORCE = {
        "rope_theta": 5000000.0,
        "partial_rotary_factor": 0.5,
        "rotary_dim": 64,
        "hidden_act": "swigluoai",
        "swiglu_alpha": 1.702,
        "swiglu_beta": 1.0,
        "swiglu_limit": 7.0,
        "head_dim": 128,
        "use_gemma_norm": True,
        "use_qk_norm": True,
        "qk_norm_type": "per_head",
        "dense_intermediate_size": 12288,
        "shared_intermediate_size": 3072,
        # CRITICAL: the official MiniMaxM3MoE creates the shared expert ONLY if
        # config.n_shared_experts is truthy (self.shared_experts=None otherwise).
        # The checkpoint config nests n_shared_experts=1 under text_config, but
        # forcing the text-only arch makes the model read the TOP-LEVEL config,
        # which lacks it -> shared expert never built -> its weights silently
        # dropped -> the shared contribution is missing from EVERY MoE layer ->
        # residual washes out to garbage. (Proven: official layer-3 moe_out ==
        # routed*2 with the shared expert, norm 338/540, entirely absent.) Force
        # it so the module is built, weights load, and the runner adds shared.
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "routed_scaling_factor": 2.0,
    }
    tc = getattr(config, "text_config", config)
    for c in {id(config): config, id(tc): tc}.values():
        for k, v in FORCE.items():
            setattr(c, k, v)
    print("[cfg] forced text values:", {k: getattr(tc, k, "<NA>") for k in
          ("rope_theta", "hidden_act", "swiglu_beta", "swiglu_alpha", "swiglu_limit",
           "partial_rotary_factor", "dense_intermediate_size")}, flush=True)
    return config


# vLLM 0.23.1 forces 'spawn' (CUDA already initialized); spawn re-imports this
# module in workers, so the LLM build MUST be guarded by __main__.
if __name__ == "__main__":
    print(f"[+{time.time()-t0:.0f}s] quant registered; building LLM (native M3)", flush=True)
    llm = LLM(
        model=CKPT,
        quantization="autoround_mixed",
        hf_overrides=_override_quant_method,
        trust_remote_code=True,
        # NOTE: PP=2 (no all-reduce) was tried to avoid the TP all-reduce issue
        # but the official M3 model lacks SupportsPP. Back to TP=2/EP=2. The
        # remaining bug is the dense-layer massive activation halving on TP=2
        # (official-model-specific: fused_allreduce_gemma_rms_norm or the gate_up
        # TP feed); M1 runs TP=2 fine, so it is not a generic all-reduce bug.
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        block_size=128,             # mandatory for MiniMax-M3 MSA sparse/index cache
        # ROOT-CAUSE FIX: FLASH_ATTN's prefill (varlen) kernel returns 0 on
        # Blackwell sm_120 for the dense layers (decode works) -> garbage. Force a
        # Blackwell-working backend for the standard Attention layers; the MSA
        # sparse layers (3-59) keep their own MiniMaxM3SparseBackend.
        # TRITON_ATTN (not FLASHINFER): the MSA sparse layers require block_size
        # 128, and FLASHINFER for the dense layers has no common block size with
        # it -> "No common block size for 128" at KV-cache init. TRITON_ATTN
        # supports 128 and is numerically correct on sm_120 (verified cos=1.0 vs
        # a full-attention reference for both dense and sparse layers).
        attention_backend=os.environ.get("M3_ATTN_BACKEND", "TRITON_ATTN"),
        max_model_len=MAXLEN,
        # Model weights are ~90.1 GiB/rank (q/k/v + fused gate_up de-quanted to
        # bf16 for the official fused-indexer/MergedColumn layers). On 95 GiB
        # cards that leaves a thin KV margin -> push utilization up and keep the
        # bring-up batch tiny. Long-context memory is optimized in a later pass
        # (keep gate_up quantized via a GPTQ split to reclaim ~1.7 GiB/rank).
        gpu_memory_utilization=0.97,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        enforce_eager=True,   # eager so M3_DIAG_HOOKS fire on the real forward
        # Blackwell (sm_120): the CUSTOM all-reduce is unreliable here (graph
        # capture breaks; SymmMem unsupported at cap 12.0). With TP=2 every layer
        # all-reduces (attention via fused_allreduce_gemma_rms_norm, MoE output),
        # so a wrong custom all-reduce corrupts everything -> deterministic
        # garbage despite correct weights. M1 is validated with NCCL all-reduce.
        disable_custom_all_reduce=True,
        dtype="bfloat16",
    )
    print(f"[+{time.time()-t0:.0f}s] LLM built; generating", flush=True)
    # Under hook-diagnostics, use ONLY the France prompt so the per-layer norms
    # match the offline oracle (dbg_oracle_layers.py, same prompt).
    # Long-context needle-in-haystack: put a fact at the START, ask at the END,
    # with >2048 tokens of filler between -> exercises the native MSA lightning
    # indexer (it must SELECT the needle's KV block from the late query). At
    # <=2048 the indexer is a no-op (full attention), so this is the real test
    # of the long-context path that ② exists for.
    if os.environ.get("M3_LONGTEST") == "1":
        needle = "The secret passphrase is crimson-falcon-8261."
        def _filler(a, b):
            return " ".join(
                f"Background note {i}: the daily weather and market summary for "
                f"entry {i} contains no important information and may be ignored."
                for i in range(a, b))
        # Needle in the MIDDLE (not block 0) -> tests the indexer's content-based
        # block SELECTION, not just always-on init/local blocks. ~7K tokens.
        long_prompt = (
            "Read the following document carefully, then answer the question.\n\n"
            f"{_filler(0, 625)}\n\n{needle}\n\n{_filler(625, 1250)}\n\n"
            "Question: What is the secret passphrase stated in the middle of the "
            "document? Answer with only the passphrase.\nAnswer:")
        ntok = len(llm.get_tokenizer()(long_prompt).input_ids)
        out = llm.generate([long_prompt], SamplingParams(temperature=0.0, max_tokens=24))
        txt = out[0].outputs[0].text
        ok = ("8261" in txt) or ("crimson" in txt.lower())
        print(f"\n==== LONG-CONTEXT MID-NEEDLE TEST (prompt {ntok} tokens) ====", flush=True)
        print(f"   -> {txt!r}\n   NEEDLE RETRIEVED: {ok}", flush=True)
        print("M3_OFFICIAL_DONE", flush=True)
        raise SystemExit(0)

    _prompts = (["The capital of France is"] if os.environ.get("M3_DIAG_HOOKS") == "1"
                else ["The capital of France is", "日本の首都は"])
    out = llm.generate(_prompts, SamplingParams(temperature=0.0, max_tokens=16))
    print("\n==== OFFICIAL M3 + our 3.2bit quant ====", flush=True)
    for o in out:
        print(f"PROMPT: {o.prompt!r}\n   -> {o.outputs[0].text!r}\n", flush=True)
    print("M3_OFFICIAL_DONE", flush=True)
