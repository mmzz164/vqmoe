#!/bin/bash
# Persistent OpenAI API server for GLM-5.2 VQ (r8) with SPARSE long-context attention on sm_120.
# ADDITIVE PRODUCTION LAUNCHER (2026-07-03): leaves start_glm_api_vq.sh (dense, ctx<=4096) UNTOUCHED
#   -> dense is one command away for rollback (see "ROLLBACK" below).
#
# Why this exists: sm_120 (RTX PRO 6000) has NO FlashMLA sparse kernel, so the dense launcher caps
#   ctx at 4096 (dense O(n^2) workspace OOMs above that). This launcher uses our env-gated GATHER
#   fallback (exact DSA semantics; branch sparse-glm-wiring, flashmla_sparse.py) to run the model's
#   native sparse-MLA path on sm_120 -> maxlen 16384 (4x) + ~15x prefill, decode unchanged (~13-16).
#
# This config == exactly what passed the S1 + S1.5 quality gates (2026-07-03): it is a copy of the
#   validated sparse_probe_serve.sh with GLM_MAXLEN default raised 8192 -> 16384 and production
#   framing. Gate evidence: needle 15/15 (3k-15k), dense-parity (<=4096), longctx-JA @9.6k 6/6 /
#   @12k 5/6, derail rate 1/24 non-systematic. See ~/work/optimize/GLM_SPARSE_SHIP_BRIEF.md.
#
# ROLLBACK to dense (ctx<=4096) in one command:
#   GLM_CKPT=/var/hf/GLM-5.2-VQ-r8/GLM-5.2-w16g128 \
#   GLM_CHAT_TEMPLATE=/var/hf/GLM-5.2-VQ-r8/GLM-5.2-w16g128/chat_template.jinja \
#   GLM_MAXLEN=4096 GLM_KV_BLOCKS=40 GLM_GPU_UTIL=0.95 GLM_LZ_PENALTY=1 GLM_BUDGET=3000 \
#   bash /var/hf/glm_serve/start_glm_api_vq.sh
#   (stop this server first: gpu_free_safe 'glm_api_server|VLLM::Worker|EngineCore')
ENV=/var/hf/envs/m3vllm-glm
export CUDA_HOME=$ENV
export LD_LIBRARY_PATH=$ENV/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/var/hf/glm_serve/pylibs:/var/hf/glm_serve:/home/mizugaihiros01/work/onecomp:$PYTHONPATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# --- SPARSE path (NOT dense): keep index_topk -> is_v32=True -> sparse MLA; gather fallback on sm120 ---
# (GLM_FORCE_DENSE intentionally NOT set)
export VLLM_TRITON_MLA_SPARSE=${VLLM_TRITON_MLA_SPARSE:-1}
export GLM_SPARSE_GATHER=${GLM_SPARSE_GATHER:-1}   # sm120 has no FlashMLA sparse kernel -> gather fallback (branch sparse-glm-wiring)
[ -n "${GLM_ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$GLM_ATTN_BACKEND"
export GLM_CKPT=${GLM_CKPT:-/var/hf/GLM-5.2-VQ-r10/GLM-5.2-w16g128}
# --- NO-THINK default (2026-07-03, user-approved reversal of the 2026-06-27 thinking-default) ---
# Why: GLM-5.2's thinking does NOT self-terminate on casual questions (measured: even "sort
# 3,1,4,1,5" ran past 600 think tokens; at ~14 tok/s every message cost minutes and free-running
# thinking occasionally collapsed into repetition loops -> the WebUI "STRICTLY ONLY..." garbage).
# The Jun-27 thinking-default existed only to dodge the no-think display quirk (answers landing in
# reasoning_content); we now fix that at the parser level instead (GLM_REASONING_PARSER=none below),
# so answers land in `content` and casual chat answers in seconds.
# Thinking opt-in per request: chat_template_kwargs {"enable_thinking": true} (nothink template
# line 118 seeds <think> open). Note: with the parser off, opt-in CoT streams raw into content
# ("CoT</think>answer") -- cosmetic; LZ+budget guardrails below still bound it.
# Old behavior (thinking default + parsed reasoning_content) in one command:
#   GLM_CHAT_TEMPLATE=/var/hf/GLM-5.2-VQ-r8/GLM-5.2-w16g128/chat_template.jinja \
#   GLM_REASONING_PARSER=deepseek_r1 bash /var/hf/glm_serve/start_glm_api_sparse.sh
export GLM_CHAT_TEMPLATE=${GLM_CHAT_TEMPLATE:-/var/hf/glm_serve/chat_template_nothink.jinja}
export GLM_MAXLEN=${GLM_MAXLEN:-16384}
# NO KV override by default: sparse attention lets vLLM auto-size KV (~225 blocks=~28k tok fit at
# 167GB weights). Only applied if the caller explicitly sets GLM_KV_BLOCKS.
[ -n "${GLM_KV_BLOCKS:-}" ] && export GLM_KV_BLOCKS
export GLM_GPU_UTIL=${GLM_GPU_UTIL:-0.97}
export GLM_PORT=${GLM_PORT:-8001}
export GLM_SERVED=${GLM_SERVED:-glm-5.2}
export GLM_REASONING_PARSER=${GLM_REASONING_PARSER:-none}  # none = answers go to content (no-think quirk fix)
export GLM_LZ_PENALTY=${GLM_LZ_PENALTY:-1}   # round-5 validated repetition-loop defense
export GLM_BUDGET=${GLM_BUDGET:-3000}        # think-budget forcing (clean </think> termination)
exec "$ENV/bin/python" /var/hf/glm_serve/glm_api_server.py "$@"
