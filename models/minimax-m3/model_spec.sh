#!/bin/bash
# model_spec.sh — MiniMax-M3 (VQ 2.4 bpw) deployment knobs.
#
# This documents the environment variables m3_vq_api_server.py honors. Every value below is the
# production default; override by exporting before launching, e.g.:
#
#     M3VQ_CKPT=/data/MiniMax-M3-VQ-2.4bit M3_PORT=8010 \
#       python m3_vq_api_server.py
#
# UNLIKE the GLM adapter, M3 serves on the OFFICIAL vLLM native MiniMax-M3 model (vLLM 0.23.1
# nightly) — the vqmoe/vllm-sm120 sparse patch is NOT used. M3's MSA lightning indexer runs on the
# stock TRITON_ATTN backend on sm_120. The only custom pieces are the OneCompression VQ kernels and
# the small key-translation adapter in this directory (see README.md).

# --- checkpoint (set to wherever you unpacked the HF download) ---
export M3VQ_CKPT=${M3VQ_CKPT:-/path/to/MiniMax-M3-VQ-2.4bit}

# --- context / memory ---
export M3_MAXLEN=${M3_MAXLEN:-40960}                 # native indexer long context
export M3_GPU_UTIL=${M3_GPU_UTIL:-0.97}              # ~65 GiB weights/GPU -> KV ~372k tokens
export M3_MAX_SEQS=${M3_MAX_SEQS:-4}

# --- execution mode ---
export M3_EAGER=${M3_EAGER:-1}                        # eager. CUDA-graph (0) hits an sm_120 capture race.
# export M3_CUDAGRAPH_MODE=PIECEWISE                  # (experimental; currently still races)

# --- thinking / reasoning ---
export M3_REASONING_PARSER=${M3_REASONING_PARSER:-minimax_m3}  # split <mm:think> into `reasoning`
# thinking mode is per-request: chat_template_kwargs {"thinking_mode": "disabled"|"adaptive"|"enabled"}
# default is "adaptive"; "disabled" gives direct answers (contains sub-2/3-bit no-exit on casual prompts)

# --- server ---
export M3_PORT=${M3_PORT:-8004}
export M3_SERVED=${M3_SERVED:-minimax-m3-vq}

# Hardware: 2x RTX PRO 6000 (sm_120), tensor-parallel=2, expert-parallel on, block_size=128
# (mandatory for the MSA sparse/index cache). ~65 GiB weights/GPU + KV.
# RAM note: the checkpoint is ~130 GiB; first load is disk-bound (~7 min from NVMe).
#
# PATHS TO EDIT before running elsewhere (these files carry machine-local absolute paths — see
# README.md "Paths you must edit"): the sys.path.insert to your OneCompression checkout and to THIS
# directory, and M3VQ_CKPT above.
