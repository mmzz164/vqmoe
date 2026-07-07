# sm_120 sparse-attention enablement

Native sparse-attention MoE models (DeepSeek-V3.2 DSA, GLM-5.2 DSA) select a **FlashMLA sparse**
attention backend in vLLM. On **sm_120 (RTX PRO 6000 / consumer Blackwell)** that backend has
**no compiled kernel** — the FlashMLA extension ships sm90/sm100 only, and the in-tree Triton
sparse suite is hard-shaped for DeepSeek-V4 geometry (448+64=512), not GLM's 576 (512 lora + 64
rope). So on sm_120 you're forced onto dense attention, which caps context at ~4096 (dense O(n²)
workspace OOMs above that after the weights are loaded).

## The fix: `flashmla_sparse.patch`

A **44-line, env-gated gather fallback** that implements the exact DSA sparse-MLA semantics with a
gather + plain attention (score over the full 576 dims, value-accumulate over the first 512;
fp32 softmax, correctness-first). Per-forward the gathered tensor is tiny (`max_num_batched_tokens`
is small on this deployment), so it is memory-safe.

Result on GLM-5.2 (sm_120, 2× RTX PRO 6000): **maxlen 16384 (4×)**, prefill ~15×, decode unchanged
(~13–16 tok/s), needle retrieval at all depths 3k–15k. Unit-tested against a dense reference
(max abs diff 3.9e-3, bf16).

## How to apply

> **Honest status — the base is NOT clean upstream.** This 44-line patch is the *last* layer of a
> larger, not-yet-published sm_120 enablement stack. The production base `prod-base-20260702`
> (`d4e0151c71`) carries additional local sm_120 patches (marlin / mla_attention / auto_gptq /
> deepseek_v2 kernels + scheduler/cudagraph audits) on top of [jasl/vllm](https://github.com/jasl/vllm),
> and the deployment uses a **precompiled** build of it (`vllm 20260622.dev0+g72261a7af`) plus
> compiled native libraries (`deep_gemm`, `flashmla` `.so`). Applying just this patch to a stock
> vLLM will **not** reproduce the deployment. Publishing the full sm_120 fork (pinned commit +
> native-lib build instructions) is a tracked follow-up — see the repo root "Reproducibility status".

Once you have the patched + built sm_120 vLLM checkout:

```bash
cd <sm120 vllm checkout>           # jasl/vllm + the prod-base sm120 stack
git apply /path/to/vqmoe/vllm-sm120/flashmla_sparse.patch   # if not already included
```

Touches only `vllm/v1/attention/backends/mla/flashmla_sparse.py`.

## How to enable

The fallback is **OFF by default** (the file behaves exactly as upstream). Turn it on with:

```bash
export GLM_SPARSE_GATHER=1
```

The GLM-5.2 launcher (`models/glm-5.2/start_glm_api_sparse.sh`) sets this for you.

## Notes / roadmap

- This is a **correctness-first fallback**, not a fused kernel — decode TPS is unchanged (the
  ~16 tok/s ceiling is the sm_120 MLA single-query decode limit, a separate issue).
- Upstream is catching up: FlashInfer #3395 / vLLM #43477 add real sm_120a sparse-MLA kernels.
  When those reach a stable wheel, this fallback can be retired in favor of the fused path.
