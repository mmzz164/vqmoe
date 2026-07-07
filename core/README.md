# core/ — model-agnostic serving core (reserved)

This directory will hold the serving pieces that are genuinely model-independent, factored out
of the per-model adapters:

- the OpenAI-compatible API entrypoint (`glm_api_server.py`),
- the vLLM engine setup (`serve_glm.py`),
- the sampling safety nets (`lz_penalty_logitproc.py` — LZ-penalty repetition defense +
  budget-forcing for clean `</think>` termination).

**It is intentionally empty in v1.** The GLM-5.2 adapter (`models/glm-5.2/`) currently carries
working, byte-identical copies of these files. We defer the extraction on purpose:

- These scripts are **live production code**. Splitting "generic core" from "GLM glue" now, with
  only one model in the repo, would be a refactor-in-the-dark that risks the working launcher for
  no immediate benefit.
- The right time to extract the core is **when model #2 arrives** — the real shared surface is
  whatever GLM and model #2 have in common, which we'll know then instead of guessing now.

Until then: to add a model, copy `models/glm-5.2/` to `models/<name>/` and swap the adapter bits
(chat template, `model_spec.sh`, attention/KV knobs, model card). The first time you do that,
lift the common files up into `core/` and have both adapters import them.
