# GLM-5.2 — VQ mixed-bit (r10, 1.90 bpw)

A **1.90 bits/weight** vector-quantized build of GLM-5.2 (744B-parameter MoE, 78 layers, 256
experts) that serves on **2× RTX PRO 6000 (sm_120)** with 16k sparse long context.

## What it is

- **Experts**: mixed 1 / 2 / 3-bit vector quantization (AQLM-style: per-group codes + shared
  codebook + GPTQ error compensation). Bit distribution `{1: 29456, 2: 12441, 3: 15703}` modules.
- **Non-expert spine** (attention / router / norm / embed / lm_head): scalar 4/8-bit.
- **Size**: experts ≈ 160 GiB, full artifact ≈ 172 GiB.
- **Allocation**: loss-aware ("gxw" = output-Fisher × input-energy × quantization error, an
  Optimal-Brain-Quantization 2nd-order cost) at a 1.90 bpw budget.

## Quality (KL to BF16, lower = better; measured via fake-quant on held-out corpora)

| | think_ja hold-out | neutral (corpus_diverse) |
|---|---|---|
| **r10 (this build, 1.90 bpw)** | **0.35155** | **0.76922** |
| r8 (predecessor, 1.85 bpw) | 0.37616 | 0.80157 |

r10 improves on r8 by **−6.5% (think_ja) / −4.0% (neutral)**, systematically across all evaluated
sequences (8/8) — a distribution-wide gain, not a single-domain artifact. Greedy arithmetic eval
22/22 terminating (JA/EN/ZH), needle retrieval verified to ~6k tokens live.

## Serving

- **Backend**: vLLM + OneCompression VQ kernels + the sm_120 sparse gather fallback
  (see `../../vllm-sm120/`). Tensor-parallel = 2.
- **Context**: 16384 (sparse). Dense fallback caps at 4096.
- **KV**: at `gpu_util 0.97`, ~25,856 tokens (≈ 1.58× concurrency at 16k).
- **Weights**: ~87.3 GiB/GPU.
- **Throughput**: prefill ~15× vs dense; decode ~13–16 tok/s (sm_120 MLA single-query ceiling).
- **Defaults**: no-think (answers in `content`; thinking is opt-in per request), LZ-penalty +
  3000-token think-budget forcing as low-bit reasoning safety nets.

Launch: `bash start_glm_api_sparse.sh` (after setting `GLM_CKPT`; see `model_spec.sh`).

## Limitations

- **Thinking mode** does not always self-terminate on casual prompts at this bit-width; the
  no-think default + budget-forcing contain this. Teacher-forcing shows BF16 terminates a runaway
  ~89% vs this build's lower rate — an inherent sub-2-bit "no-exit attractor", mitigated at serving.
- **Decode ~16 tok/s** is a hardware ceiling (sm_120 MLA decode is single-query), not tunable here.
- Long context > 16k and multi-query speculative decode are not available on sm_120 with the
  current kernels.

## Provenance & license

- Base model: GLM-5.2 (Apache-2.0). Quantizer: [OneCompression](https://github.com/mmzz164/OneCompression) (MIT).
- This build's serving config is in `vqmoe` (intended Apache-2.0).
- The 1.4 TB BF16 original was released after this build; re-quantization requires re-downloading it.
