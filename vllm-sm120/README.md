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

Base commit (jasl/vllm production snapshot): **`d4e0151c71` (prod-base-20260702)**.

```bash
cd <your vllm checkout>            # jasl/vllm at/near d4e0151c71
git apply /path/to/vqmoe/vllm-sm120/flashmla_sparse.patch
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
