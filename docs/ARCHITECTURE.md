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

## Note on M3 (model #2): the middle layer is optional

The three-layer diagram above assumes the model needs `vqmoe/vllm-sm120` to run sparse attention on
sm_120. That is true for **GLM-5.2** (DSA has no sm_120 sparse kernel). It is **not** true for
**MiniMax-M3**: upstream vLLM 0.23.1 ships native MiniMax-M3 with the MSA lightning indexer, which
runs on the stock `TRITON_ATTN` backend on sm_120. So for M3 the middle "make it run on cheap GPUs"
layer collapses to "use official vLLM", and its adapter (`models/minimax-m3/`) is pure
key-translation + quant dispatch on top of the stock wheel. The "quantize" and "serve well" layers
are unchanged. Which middle layer you need is a per-model property, not a framework constant.

## Before publishing (checklist)

- [x] Add a top-level `LICENSE` (MIT) and a `NOTICE` crediting OneCompression (MIT, Fujitsu + mmzz164),
      GLM-5.2 (MIT, zai-org), MiniMax-M3 (MiniMax Community License), vLLM (Apache-2.0), jasl/vllm.
- [x] Fill the HF checkpoint URLs in `README.md` / per-model `MODEL_CARD.md` (GLM + M3 uploaded).
- [x] Machine-local absolute paths: kept as example defaults; GLM's are overridable env vars, M3's
      are listed under `models/minimax-m3/README.md` "Paths you must edit".
- [x] Public vs private: public (owner-confirmed).
