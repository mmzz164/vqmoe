#!/bin/bash
# model_spec.sh — GLM-5.2 (VQ r10, 1.90 bpw) deployment knobs.
#
# This documents the interface the launcher (start_glm_api_sparse.sh) exposes as environment
# variables. Every value below is the production default baked into that launcher; override any of
# them by exporting before calling the launcher, e.g.:
#
#     GLM_CKPT=/data/GLM-5.2-VQ-r10/GLM-5.2-w16g128 GLM_PORT=8010 bash start_glm_api_sparse.sh
#
# The launcher is copied VERBATIM from production and is NOT parameterized beyond these env vars.

# --- checkpoint (set this to wherever you unpacked the HF download) ---
export GLM_CKPT=${GLM_CKPT:-/path/to/GLM-5.2-VQ-r10/GLM-5.2-w16g128}

# --- sparse long context on sm_120 (see ../../vllm-sm120/) ---
export GLM_SPARSE_GATHER=${GLM_SPARSE_GATHER:-1}     # gather fallback ON (no FlashMLA sparse kernel on sm120)
export GLM_MAXLEN=${GLM_MAXLEN:-16384}               # 16k context (4x the dense cap)
export GLM_GPU_UTIL=${GLM_GPU_UTIL:-0.97}            # r10 weights ~87.3 GiB/GPU -> 0.97 leaves KV ~25.9k tok

# --- chat / thinking behaviour ---
export GLM_CHAT_TEMPLATE=${GLM_CHAT_TEMPLATE:-$(dirname "$0")/chat_template_nothink.jinja}
export GLM_REASONING_PARSER=${GLM_REASONING_PARSER:-none}  # answers land in `content` (no-think default)
# thinking is opt-in per request: chat_template_kwargs {"enable_thinking": true}

# --- low-bit reasoning safety nets (keep these on) ---
export GLM_LZ_PENALTY=${GLM_LZ_PENALTY:-1}           # repetition-loop defense
export GLM_BUDGET=${GLM_BUDGET:-3000}                # think-budget forcing -> clean </think>

# --- server ---
export GLM_PORT=${GLM_PORT:-8001}
export GLM_SERVED=${GLM_SERVED:-glm-5.2}

# Hardware: 2x RTX PRO 6000 (sm_120), tensor-parallel=2. ~87.3 GiB weights/GPU + KV.
# RAM note: the checkpoint is ~172 GiB; with <checkpoint-size RAM, first load is disk-bound
# (~7 min from NVMe, far slower from a spinning disk).
