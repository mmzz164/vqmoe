# core/ — model-agnostic serving core (reserved)

This directory will hold the serving pieces that are genuinely model-independent, factored out
of the per-model adapters:

- the OpenAI-compatible API entrypoint (`glm_api_server.py`),
- the vLLM engine setup (`serve_glm.py`),
- the sampling safety nets (`lz_penalty_logitproc.py` — LZ-penalty repetition defense +
  budget-forcing for clean `</think>` termination).

**It is intentionally empty — and now, after model #2, deliberately so.** The plan was to extract
a shared core once a second model arrived and the real overlap was known. Model #2
(`models/minimax-m3/`) has landed, and the finding is: **the concrete overlap is thin.**

- GLM-5.2 serves on a **patched** vLLM (out-of-tree model + the `vllm-sm120/` sparse gather
  fallback); MiniMax-M3 serves on **official** vLLM's native model. Different substrate → different
  engine setup, different loader, different server entrypoint.
- What they genuinely share — the VQ dequant kernels, the OpenAI API *shape*, the low-bit sampling
  concerns — already lives in OneCompression and vLLM, not in glue we'd hoist here.

So a "generic core" module would be indirection over two adapters that have little literal code in
common. Each adapter stays self-contained. To add a model, copy the *closest* existing adapter
(`models/glm-5.2/` for DSA/patched-vLLM models, `models/minimax-m3/` for natively-supported ones)
and swap the adapter bits (chat template, `model_spec.sh`, attention/KV knobs, key translation,
model card). Revisit extraction only if a third model makes a real shared surface obvious.
