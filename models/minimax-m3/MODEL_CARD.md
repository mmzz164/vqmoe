# MiniMax-M3 — VQ mixed-bit (2.4 bpw)

A **2.4 bits/weight** vector-quantized build of MiniMax-M3 (427B-parameter MoE, 57 sparse-MoE
layers × 128 experts, top-4 routing) that serves on **2× RTX PRO 6000 (sm_120)** through
**official vLLM** — no sm_120 sparse patch required.

HF checkpoint: **[aquaman164/MiniMax-M3-VQ-2.4bit](https://huggingface.co/aquaman164/MiniMax-M3-VQ-2.4bit)**

## What it is

- **Routed experts** (413B, 96.7% of the model): mixed **1 / 2 / 3-bit** vector quantization
  (AQLM-style: per-group codes + shared codebook + GPTQ error compensation). Module bit
  distribution `{1: 6630, 2: 2995, 3: 12263}`.
- **Non-expert spine** (attention / shared experts / dense MLP): symmetric 4-bit RTN. Router, MSA
  indexer, norms, embeddings, lm_head, and the **vision tower stay bf16**.
- **Size**: ~130.2 GiB (46,484 tensors) — 16% of the 796 GiB BF16 original, ~65 GiB/GPU under TP=2.
- **Allocation**: loss-aware ("gxw" = output-Fisher × input-energy × quantization error, an
  Optimal-Brain-Quantization 2nd-order cost) at a 2.4 bpw budget.

## Quality (KL to BF16, lower = better; fake-quant on held-out corpora)

| corpus | KL(BF16 ‖ this build) | PPL BF16 → 2.4-bit |
|---|---|---|
| Japanese-reasoning hold-out | **0.219** | 11.51 → 12.14 |
| neutral multilingual (diverse) | **0.440** | 2.90 → 3.84 |

Same per-token degradation (~0.28 nats/token) as the same model's prior **3.2-bit** AutoRound build,
at **0.8 fewer bits/weight** — VQ + compensation + loss-aware allocation holds quality further down
the bit curve than scalar. Greedy arithmetic terminates in JA/EN/ZH; needle retrieval verified to
~15k tokens.

## Serving

- **Backend**: official vLLM 0.23.1 native MiniMax-M3 (MSA lightning indexer, `TRITON_ATTN`) +
  OneCompression VQ kernels. **The `vqmoe/vllm-sm120/` sparse patch is NOT used here** — M3's native
  indexer runs on stock sm_120 kernels. TP=2, expert-parallel on.
- **Context**: 40,960. **KV** at `gpu_util 0.97` ≈ 372,736 tokens (~27 GiB/GPU free — small weight
  footprint = generous headroom).
- **Throughput**: ~7.3 tok/s decode (eager). CUDA-graph capture hits an sm_120 race and is
  disabled; eager is the supported path.
- **Thinking**: native `<mm:think>` blocks. `chat_template_kwargs {"thinking_mode": "disabled"}` →
  direct answers (default `adaptive`). The server sets `--reasoning-parser minimax_m3` so reasoning
  lands in the `reasoning` field, content clean.

Launch: `bash model_spec.sh` documents the knobs; then run `m3_vq_api_server.py` (see
[`README.md`](README.md) for the paths you must edit first).

## Limitations

- **Vision** weights retained (bf16) but the served path is **text-only**; image-text is unverified
  for this build.
- Reasoning can fail to self-terminate on casual creative prompts (sub-2/3-bit "no-exit"); the
  `thinking_mode: "disabled"` toggle contains it.
- Decode ~7 tok/s (eager) — CUDA-graph blocked by an sm_120 capture race, not tunable here yet.

## Provenance & license

- Base model: [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3), **MiniMax Community
  License** (non-MIT). This is a quantized derivative and inherits it — **commercial use requires
  displaying "Built with MiniMax M3"** and, above the license's revenue threshold, prior
  authorization from MiniMax. Read the checkpoint's `LICENSE` before commercial deployment.
- Quantizer: [OneCompression](https://github.com/mmzz164/OneCompression) (MIT, Fujitsu Ltd. + VQ
  extensions © mmzz164). Serving glue (this repo): MIT.
