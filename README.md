# vqmoe — serve sub-2-bit VQ-quantized giant MoE models on consumer GPUs

`vqmoe` is a **serving toolkit** for very large Mixture-of-Experts LLMs that have been
quantized to **sub-2-bit** with vector quantization (VQ / AQLM-style), running on a small
number of **consumer Blackwell (sm_120, e.g. RTX PRO 6000)** GPUs.

It is the *deployment* half of the story. The *quantization* half — the VQ mixed-bit
quantizer, the AutoBit/loss-aware bit allocation, and the vLLM GPTQ/VQ dequant kernels — lives in
[OneCompression](https://github.com/mmzz164/OneCompression). `vqmoe` takes those quantized
checkpoints and makes them **serve well**: sparse long context on hardware that has no sparse
kernel, an OpenAI-compatible API, and the sampling safety nets that keep low-bit reasoning models
from running away.

## Why it exists

Off-the-shelf stacks can't run these models here:

- **llama.cpp / GGUF** have no VQ MoE kernel → a 744B model at 1.9 bits/weight simply won't load.
- **Stock vLLM** on sm_120 has **no FlashMLA sparse kernel**, so native sparse-attention models
  (DeepSeek-V3.2 / GLM-5.2-style DSA) are capped at short context by dense O(n²) workspace.

`vqmoe` closes both gaps with a small, additive layer on top of vLLM + OneCompression.

## What's model-agnostic vs model-specific

| Shared (the framework) | Per-model (a thin adapter under `models/`) |
|---|---|
| sm_120 sparse-attention enablement (`vllm-sm120/`) | bitmap / quantization config (in OneCompression + the HF checkpoint) |
| OpenAI-compatible API server + launcher pattern | chat template |
| no-think default, LZ-penalty + budget-forcing safety nets | attention geometry, KV / gpu-util tuning |
| | model card |

**Model #1 is GLM-5.2** (`models/glm-5.2/`). Adding a model = dropping in a new adapter; the
shared serving core is factored out of the GLM adapter once model #2 lands (see `core/README.md`).

## Layout

```
vqmoe/
├── vllm-sm120/       # sparse-attention-on-sm120 enablement (shared): the gather-fallback patch
├── core/             # model-agnostic serving core (extracted when model #2 arrives)
├── models/
│   └── glm-5.2/      # model #1: launcher + server + chat template + MODEL_CARD
└── docs/ARCHITECTURE.md
```

## Quick start (GLM-5.2)

1. Get the quantized checkpoint (HF: TBD) and the [OneCompression](https://github.com/mmzz164/OneCompression) VQ kernels.
2. Apply the sm_120 sparse patch to a compatible vLLM checkout — see [`vllm-sm120/README.md`](vllm-sm120/README.md).
3. Serve — see [`models/glm-5.2/`](models/glm-5.2/) (`start_glm_api_sparse.sh`, `MODEL_CARD.md`).

## Dependencies (referenced, not vendored)

- **[OneCompression](https://github.com/mmzz164/OneCompression)** @ tag `glm-serving-v1` — VQ MoE
  quantizer + vLLM dequant kernels.
- **vLLM** with the sm_120 patch stack (base: [jasl/vllm](https://github.com/jasl/vllm)) — see `vllm-sm120/`.
- Native libs `deep_gemm`, `flashmla` (compiled `.so`, build artifacts) — **not** vendored here.

Pinned environment (the deployment): `python 3.10`, `torch 2.11.0`, `transformers 5.12.0`
(GLM-5.2 needs ≥5.12), `triton 3.6.0`, `vllm 20260622.dev0+g72261a7af` (the precompiled sm_120 fork).

## Reproducibility status (honest)

This v1 is a **faithful record of the serving layer**, not yet a turn-key clone. What's covered vs
what's still needed for "anyone can run it the same way":

| Piece | Status |
|---|---|
| Serving glue (launcher, API server, safety nets, chat template) | ✅ in this repo |
| VQ dequant kernels | ✅ public (OneCompression @ `glm-serving-v1`) |
| sm_120 sparse gather fallback | ✅ patch here, but on top of ↓ |
| **sm_120 vLLM base (patch stack + compiled native libs)** | ❌ **not yet published** — the hard gap |
| Quantized checkpoint | ⏳ HF upload pending |
| Pinned env / build recipe | ⏳ versions listed above; no one-shot installer yet |

The load-bearing gap is the **sm_120 vLLM fork** (a substantial, hand-patched + hand-compiled
stack). Making this repo truly reproducible means publishing that fork at a pinned commit with
native-lib build instructions — tracked as the next step, not done here.

## License

The serving glue here is original and permissive (MIT or Apache-2.0 — TBD, added as a `LICENSE`
file on first publication). Everything it builds on is permissive too:
GLM-5.2 (**MIT**, zai-org), OneCompression (**MIT**, Fujitsu + mmzz164), vLLM (**Apache-2.0**).
See the "before publishing" checklist in `docs/ARCHITECTURE.md`.
