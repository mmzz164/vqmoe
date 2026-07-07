# Architecture

## Three layers, two repos

Serving a `vqmoe` model composes three layers. Two of them are shared infrastructure; only the
thin top layer is per-model.

```
                         ┌─────────────────────────────────────────────┐
  make it small          │  OneCompression  (separate repo, MIT)        │
  (quantize)             │  · VQ mixed-bit MoE quantizer (AQLM-style)   │
                         │  · loss-aware bit allocation (AutoBit/gxw)   │
                         │  · vLLM GPTQ/VQ dequant kernels              │
                         └───────────────────────┬─────────────────────┘
                                                 │ quantized checkpoint + kernels
                         ┌───────────────────────▼─────────────────────┐
  make it run on         │  vqmoe/vllm-sm120  (this repo)               │
  cheap GPUs             │  · sparse-attention gather fallback for      │
                         │    sm_120 (no FlashMLA sparse kernel there)  │
                         └───────────────────────┬─────────────────────┘
                                                 │ vLLM that serves sparse long-ctx on sm120
                         ┌───────────────────────▼─────────────────────┐
  make it serve well     │  vqmoe/models/<model>  (this repo)           │
  (per model)            │  · OpenAI API server + launcher              │
                         │  · chat template, KV/util, safety nets       │
                         └─────────────────────────────────────────────┘
```

- **OneCompression** = "make it small". It owns the quantization math and the dequant kernels that
  vLLM calls at inference. `vqmoe` depends on it but does not duplicate it.
- **`vqmoe/vllm-sm120`** = "make it run on cheap GPUs". One small patch that unlocks native sparse
  attention on consumer Blackwell. Shared by every model.
- **`vqmoe/models/<model>`** = "make it serve well". The launcher, the OpenAI server, the sampling
  safety nets, and the model-specific knobs (chat template, context, KV, util).

## The v1 shortcut (and why)

In v1 the model-agnostic serving code (API server, engine setup, logits processors) physically
lives inside `models/glm-5.2/`, not in `core/`. That's deliberate: with a single model in the
repo, factoring a "generic core" out of live production code would be a guess with downside and no
upside. `core/` is extracted when model #2 arrives and the real shared surface is known. See
`core/README.md`.

## Before publishing (checklist)

- [ ] Add a top-level `LICENSE` (intended Apache-2.0) and a `NOTICE` crediting OneCompression (MIT),
      GLM-5.2 (Apache-2.0), vLLM (Apache-2.0), jasl/vllm.
- [ ] Fill the HF checkpoint URL in `README.md` / `MODEL_CARD.md` once the model is uploaded.
- [ ] Sanity-scan for machine-local absolute paths — the launcher's defaults point at `/var/hf/...`
      but every one is an overridable env var (documented in `model_spec.sh`); decide whether to
      keep them as example defaults or blank them.
- [ ] Confirm public vs private with the owner.
