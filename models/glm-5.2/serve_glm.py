# SPDX-License-Identifier: Apache-2.0
"""Serve our GLM-5.2 AutoBit fit artifact (180.82 GiB, mixed 1/2/3-bit experts +
asym-MSE 4-bit spine) on the NATIVE vLLM GlmMoeDsaForCausalLM model (vLLM 0.23.1,
deepseek_v2 backbone: MLA + DSA sparse indexer + MoE). vLLM unchanged; we only
register the quantization ("autoround_mixed") and rewrite quant_method.

Run in m3vllm env:
  /home/mizugaihiros01/anaconda3/envs/m3vllm/bin/python /var/hf/glm_serve/serve_glm.py

Bring-up env knobs:
  GLM_TP (default 2), GLM_EP (0/1, default 0), GLM_MAXLEN (default 2048),
  GLM_GPU_UTIL (default 0.95), GLM_ATTN (default TRITON_ATTN), GLM_EAGER (default 1)
"""
import os, sys, time

# 1-bit experts -> grouped Triton path raises; must use the eager dequant path.
os.environ.setdefault("MIXED_MOE_GROUPED", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, "/var/hf/glm_serve")

CKPT = os.environ.get("GLM_CKPT", "/var/hf/GLM-5.2-Smart-AutoBit-1bit/GLM-5.2-w16g128")
TP = int(os.environ.get("GLM_TP", "2"))
EP = os.environ.get("GLM_EP", "0") == "1"
MAXLEN = int(os.environ.get("GLM_MAXLEN", "2048"))
GPU_UTIL = float(os.environ.get("GLM_GPU_UTIL", "0.95"))
# Default: let vLLM auto-select per-layer (dense MLA layers 0-2 -> FLASH_ATTN_MLA,
# DSA sparse layers 3-77 -> FLASHMLA_SPARSE). TRITON_ATTN supports neither MLA nor
# sparse, so do NOT force it (that was the M3 path). Override via GLM_ATTN if needed.
ATTN = os.environ.get("GLM_ATTN", "")
EAGER = os.environ.get("GLM_EAGER", "1") == "1"
t0 = time.time()

import glm_quant   # noqa: registers "autoround_mixed" + GLM _FUSED_TO_CONSTITUENTS
from vllm import LLM, SamplingParams


def _override_quant_method(config):
    # Tell vLLM to use our registered mixed-bit GPTQ config for this checkpoint.
    qc = getattr(config, "quantization_config", None)
    if isinstance(qc, dict) and qc.get("quant_method") == "auto-round":
        qc["quant_method"] = "autoround_mixed"
    # GLM-5.2 native arch reads a COMPLETE config (architectures GlmMoeDsaForCausalLM,
    # all MLA/DSA/MoE keys present) -> no FORCE dict needed (unlike the M3 text-only port).
    # DSA sparse attention is disabled (-> dense MLA) by the deepseek_v2 hasattr patch
    # in glm_quant.py (sm_120 has no FlashMLA-Sparse kernels); nothing to force here.
    return config


if __name__ == "__main__":
    print(f"[+{time.time()-t0:.0f}s] quant registered; building LLM (native GLM-5.2) "
          f"TP={TP} EP={EP} MAXLEN={MAXLEN} util={GPU_UTIL} attn={ATTN} eager={EAGER} "
          f"grouped={os.environ['MIXED_MOE_GROUPED']}", flush=True)
    kw = dict(
        model=CKPT,
        quantization="autoround_mixed",
        hf_overrides=_override_quant_method,
        trust_remote_code=True,
        tensor_parallel_size=TP,
        enable_expert_parallel=EP,         # EP=0 first: test eager+TP expert sharding
        block_size=128,                    # DSA sparse/index cache
        max_model_len=MAXLEN,
        gpu_memory_utilization=GPU_UTIL,
        max_num_seqs=1,
        max_num_batched_tokens=int(os.environ.get("GLM_MAXBATCH", "512")),
        enforce_eager=EAGER,
        # sm_120 custom all-reduce was flagged unreliable -> default off. GLM_DISABLE_CUSTOM_AR=0
        # re-enables it (custom AR is much faster than NCCL ring for small decode messages; the
        # EP all-reduce is the top decode cost). Verify output correctness when enabling.
        disable_custom_all_reduce=os.environ.get("GLM_DISABLE_CUSTOM_AR", "1") == "1",
        dtype="bfloat16",
    )
    if ATTN:
        kw["attention_backend"] = ATTN
    # Cap KV cache blocks so the eager MoE forward keeps physical headroom.
    # Weights fill ~98% of the per-GPU budget; if vLLM grabs all remaining
    # memory for KV, the forward pass OOMs (esp. on the heavier EP rank). With
    # block_size=128, e.g. GLM_KV_BLOCKS=20 -> 2,560 tokens (>= MAXLEN 2048).
    _kvblocks = os.environ.get("GLM_KV_BLOCKS", "")
    if _kvblocks:
        kw["num_gpu_blocks_override"] = int(_kvblocks)
    llm = LLM(**kw)
    print(f"[+{time.time()-t0:.0f}s] LLM built; generating", flush=True)
    _pf = os.environ.get("GLM_PROMPTS_FILE", "")
    if _pf:
        prompts = [l.rstrip("\n") for l in open(_pf, encoding="utf-8") if l.strip()]
    else:
        prompts = ["The capital of France is", "日本の首都は"]
    out = llm.generate(prompts, SamplingParams(
        temperature=0.0, max_tokens=int(os.environ.get("GLM_PROMPT_TOK", "16"))))
    print("\n==== GLM-5.2 AutoBit fit + native vLLM ====", flush=True)
    for o in out:
        print(f"PROMPT: {o.prompt!r}\n   -> {o.outputs[0].text!r}\n", flush=True)
    # Steady-state decode TPS: the gen above warmed up Triton JIT, so this timed
    # single-prompt run (default 64 new tokens) measures decode without warmup.
    bench_tok = int(os.environ.get("GLM_MAXTOK", "64"))
    _tb = time.time()
    bout = llm.generate(["The capital of France is"],
                        SamplingParams(temperature=0.0, max_tokens=bench_tok))
    _dt = time.time() - _tb
    _n = len(bout[0].outputs[0].token_ids)
    print(f"BENCH: {_n} tok in {_dt:.2f}s = {_n/_dt:.2f} tok/s "
          f"(steady-state decode, frugal={os.environ.get('MIXED_MOE_FRUGAL','0')} "
          f"grouped={os.environ['MIXED_MOE_GROUPED']} eager={EAGER})", flush=True)
    print("GLM_SERVE_DONE", flush=True)
