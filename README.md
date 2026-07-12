# vqmoe — serve sub-2-bit VQ-quantized giant MoE models on consumer GPUs

`vqmoe` is a **serving toolkit** for Mixture-of-Experts LLMs that have been
quantized to **~2 bits/weight and below** with vector quantization (VQ / AQLM-style), running on a
small number of **consumer Blackwell (sm_120, e.g. RTX PRO 6000)** GPUs — and, since model #3,
on **Apple Silicon (MLX + custom Metal kernels)**.

It is the *deployment* half of the story. The *quantization* half — the VQ mixed-bit
quantizer, the loss-aware bit allocation, and the vLLM GPTQ/VQ dequant kernels — lives in
[OneCompression](https://github.com/mmzz164/OneCompression). `vqmoe` takes those quantized
checkpoints and makes them **serve well**: long context on hardware with no off-the-shelf path, an
OpenAI-compatible API, and the sampling considerations that keep low-bit reasoning models usable.

## Why it exists

Off-the-shelf stacks can't run these models here:

- **llama.cpp / GGUF** have no VQ MoE kernel → a 744B model at 1.9 bits/weight simply won't load.
- **Stock vLLM** on sm_120 has **no FlashMLA sparse kernel**, so DSA-style native sparse-attention
  models (DeepSeek-V3.2 / GLM-5.2) are capped at short context by dense O(n²) workspace — until you
  add the gather fallback in `vllm-sm120/`.

`vqmoe` closes these gaps with a small, additive layer on top of vLLM + OneCompression.

## Models

| # | model | bpw | size | serving substrate |
|---|---|---|---|---|
| 1 | [GLM-5.2](models/glm-5.2/) (744B) | 1.90 | 172 GiB | **patched** vLLM + `vllm-sm120/` sparse gather (DSA has no sm_120 kernel) |
| 2 | [MiniMax-M3](models/minimax-m3/) (427B) | 2.40 | 130 GiB | **official** vLLM 0.23.1 native M3 (MSA indexer runs on stock kernels — no patch) |
| 3 | [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b-mlx/) (35B) | 2.0 / 2.38 / 3.13 (experts) | 10.5 / 11.5 / 15.0 GiB | **MLX on Apple Silicon** — custom `mx.fast.metal_kernel` VQ GEMV (no CUDA at all) |

The three differ in serving substrate, and that difference is the interesting part: GLM-5.2's
DSA sparse attention has no sm_120 kernel and needs the hand-written gather fallback here; MiniMax-M3's
MSA lightning indexer is supported natively by upstream vLLM, so its adapter is pure key-translation
+ quant dispatch on top of the stock wheel; Qwen3.6 leaves CUDA entirely — the same VQ math ported to
MLX with ~150 lines of Metal, putting a 35B MoE at scalar-beating quality on a 16 GB MacBook.
Adding a model = dropping a new adapter under `models/`.

## Layout

```
vqmoe/
├── vllm-sm120/       # sparse-attention-on-sm120 gather fallback (used by GLM, NOT by M3)
├── core/             # model-agnostic serving core (still deferred — see below)
├── models/
│   ├── glm-5.2/              # model #1: patched-vLLM launcher + server + chat template + MODEL_CARD
│   ├── minimax-m3/           # model #2: official-vLLM VQ server + key-translation adapter + MODEL_CARD
│   └── qwen3.6-35b-a3b-mlx/  # model #3: MLX/Apple Silicon — VQ Metal kernels + loader + server
└── docs/ARCHITECTURE.md
```

`core/` stays intentionally empty: now that model #2 has landed, the concrete overlap between the two
adapters turns out to be **thin** — they sit on different vLLM substrates (patched vs official) with
different loaders and servers. What they truly share (the VQ dequant kernels, the OpenAI API shape,
the low-bit sampling concerns) is small and already lives in OneCompression + vLLM, so forcing a
shared "core" module would be more indirection than reuse. See `core/README.md`.

## Quick start

Pick the adapter for your model and follow its README:

- **GLM-5.2** — [`models/glm-5.2/`](models/glm-5.2/): needs the `vllm-sm120/` sparse patch, then
  `start_glm_api_sparse.sh`. Checkpoint: [aquaman164/GLM-5.2-VQ-1.9bit](https://huggingface.co/aquaman164/GLM-5.2-VQ-1.9bit).
- **MiniMax-M3** — [`models/minimax-m3/`](models/minimax-m3/): stock official vLLM 0.23.1 + the
  adapter; no patch. Checkpoint: [aquaman164/MiniMax-M3-VQ-2.4bit](https://huggingface.co/aquaman164/MiniMax-M3-VQ-2.4bit).
- **Qwen3.6-35B-A3B (Mac)** — [`models/qwen3.6-35b-a3b-mlx/`](models/qwen3.6-35b-a3b-mlx/):
  `pip install mlx mlx-lm`, download, `sh run_mac.sh <model_dir>`. No CUDA, no OneCompression at
  runtime. Checkpoints: [VQ-2.4bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.4bpw) (10.5 GiB, d=8 1.5-bit tier — fits 16 GB Macs with headroom),
  [VQ-2.6bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.6bpw) (11.5 GiB)
  and [VQ-3.4bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-3.4bpw) (15.0 GiB, beats scalar 3.5bpw on quality).

The two vLLM models need the [OneCompression](https://github.com/mmzz164/OneCompression) VQ dequant
kernels + shared codebooks on `PYTHONPATH`; the MLX model is self-contained (kernels bundled).

## Dependencies (referenced, not vendored)

- **[OneCompression](https://github.com/mmzz164/OneCompression)** — VQ MoE quantizer + vLLM dequant
  kernels + the shared codebooks. (GLM serving pinned at tag `glm-serving-v1`.)
- **vLLM**:
  - GLM-5.2 → the sm_120 **patch stack** (base: [jasl/vllm](https://github.com/jasl/vllm)) + compiled
    native libs `deep_gemm`, `flashmla` — see `vllm-sm120/`. **Not vendored** (the hard gap below).
  - MiniMax-M3 → **official** vLLM 0.23.1 nightly (native MiniMax-M3). No fork.
- `transformers 5.12.0` (both models need ≥5.12), `torch 2.11.0`.

## Reproducibility status (honest)

A faithful record of the serving layer. What's covered vs still needed, **per model**:

| Piece | GLM-5.2 | MiniMax-M3 |
|---|---|---|
| Adapter (server, launcher/loader, safety nets) | ✅ in this repo | ✅ in this repo |
| VQ dequant kernels + codebooks | ✅ OneCompression | ✅ OneCompression |
| Quantized checkpoint | ✅ [HF](https://huggingface.co/aquaman164/GLM-5.2-VQ-1.9bit) | ✅ [HF](https://huggingface.co/aquaman164/MiniMax-M3-VQ-2.4bit) |
| Base vLLM | ❌ **sm_120 fork not yet published** — the hard gap | ✅ **official** 0.23.1 nightly, no patch |
| Machine-local paths / one-shot installer | ⚠ overridable env defaults; no lockfile | ⚠ four paths to edit; no lockfile |

The load-bearing gap is **GLM's sm_120 vLLM fork** (a substantial hand-patched + hand-compiled stack);
publishing it at a pinned commit with build instructions is the next step. **MiniMax-M3 has no such
gap** — official vLLM + OneCompression + the adapter here is the whole stack.

## License

Serving glue here is **MIT** (see `LICENSE` / `NOTICE`). What it builds on:
- GLM-5.2 — **MIT** (zai-org).
- MiniMax-M3 — **MiniMax Community License** (non-MIT; commercial use requires "Built with MiniMax M3"
  attribution and, above a revenue threshold, prior authorization). The M3 checkpoint carries the
  license; read it before commercial deployment.
- OneCompression — **MIT** (Fujitsu Ltd. + mmzz164). vLLM — **Apache-2.0**.
