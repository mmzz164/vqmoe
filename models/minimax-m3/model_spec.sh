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

# --- parallelism (the ~130 GiB weights need enough GPUs to hold TP shards + KV) ---
export M3_TP=${M3_TP:-2}                              # tensor-parallel size. 2x RTX PRO 6000 -> 2.
                                                     # 8x RTX 3090 (24 GB) -> 8 (weights don't fit <6).
export M3_PP=${M3_PP:-1}                              # pipeline-parallel size
export M3_EXPERT_PARALLEL=${M3_EXPERT_PARALLEL:-1}    # expert parallelism (128 experts split across ranks)
export M3_DISABLE_CUSTOM_AR=${M3_DISABLE_CUSTOM_AR:-1}  # custom all-reduce OFF (needed on Blackwell sm_120;
                                                     # Ampere w/ NVLink may benefit from 0 — try it)

# --- context / memory / batching ---
export M3_MAXLEN=${M3_MAXLEN:-40960}                 # native indexer long context
export M3_GPU_UTIL=${M3_GPU_UTIL:-0.97}              # ~65 GiB weights/GPU @ TP=2 -> KV ~372k tokens
export M3_MAX_SEQS=${M3_MAX_SEQS:-4}
export M3_MAX_BATCHED_TOKENS=${M3_MAX_BATCHED_TOKENS:-2048}  # prefill batch; raise (4096+) for throughput

# --- execution mode ---
export M3_EAGER=${M3_EAGER:-1}                        # eager. On sm_120 (Blackwell) CUDA-graph capture
                                                     # races -> keep 1. On Ampere/Ada it works: set 0
                                                     # (+ M3_CUDAGRAPH_WARMUPS=2) for ~2-3x decode.
export M3_CUDAGRAPH_WARMUPS=${M3_CUDAGRAPH_WARMUPS:-2}  # eager warmups before capture (primes the VQ
                                                     # codebook cache; needed when M3_EAGER=0)
# export M3_CUDAGRAPH_MODE=PIECEWISE                  # (override cudagraph mode if the default misbehaves)

# --- thinking / reasoning ---
export M3_REASONING_PARSER=${M3_REASONING_PARSER:-minimax_m3}  # split <mm:think> into `reasoning`
# thinking mode is per-request: chat_template_kwargs {"thinking_mode": "disabled"|"adaptive"|"enabled"}
# default is "adaptive"; "disabled" gives direct answers (contains sub-2/3-bit no-exit on casual prompts)

# --- server ---
export M3_PORT=${M3_PORT:-8004}
export M3_SERVED=${M3_SERVED:-minimax-m3-vq}

# Reference configs:
#   2x RTX PRO 6000 (sm_120, 96 GB): M3_TP=2 M3_EAGER=1  -> ~7 tok/s (eager; CUDA-graph blocked here)
#   8x RTX 3090     (sm_86, 24 GB):  M3_TP=8 M3_EAGER=0 M3_CUDAGRAPH_WARMUPS=2 M3_MAX_BATCHED_TOKENS=4096
#                                     -> ~20 tok/s (community-reported; CUDA graphs work off-Blackwell)
# block_size stays 128 (mandatory for the MSA sparse/index cache — a correctness requirement).
# RAM note: the checkpoint is ~130 GiB; first load is disk-bound (~7 min from NVMe).
#
# PATHS TO EDIT before running elsewhere (these files carry machine-local absolute paths — see
# README.md "Paths you must edit"): the sys.path.insert to your OneCompression checkout and to THIS
# directory, and M3VQ_CKPT above.
