# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for MiniMax-M3 VQ (2.4bpw, r1) on the OFFICIAL
native vLLM M3 (0.23.1) — the VQ sibling of m3_official_api_server.py (②).

Reuses the entire ② stack unchanged (serve_m3_official._override_quant_method
incl. the FORCE dict + n_shared_experts=1, m3_official_loader key translation,
m3_quant's qkv/gate_up handling) and adds only:
  - quantization="autoround_mixed_vq" (m3_quant_vq: parser keeps the per-expert
    format:"vq" marker -> dispatch reaches MixedVQMoEMethod / vq kernels)
  - the VQ checkpoint path (~127 GiB vs 3.2bit's ~180: weights ~64 GiB/rank ->
    much larger KV headroom at the same util)

Start (m3vllm env; M3_OFFICIAL_PORT is set automatically):
  /home/mizugaihiros01/anaconda3/envs/m3vllm/bin/python /var/hf/vllm_m3/m3_vq_api_server.py
Test:
  curl http://localhost:8004/v1/models
  curl http://localhost:8004/v1/chat/completions -H 'Content-Type: application/json' \
       -d '{"model":"minimax-m3-vq","messages":[{"role":"user","content":"日本の首都は?"}],"max_tokens":30}'
"""
import os
import sys
import asyncio

os.environ["M3_OFFICIAL_PORT"] = "1"   # gate m3_quant's gate_up/qkv handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("MIXED_MOE_GROUPED", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, "/var/hf/vllm_m3")

CKPT = os.environ.get("M3VQ_CKPT", "/var/hf/MiniMax-M3-VQ-r1/MiniMax-M3-w16g128")
PORT = os.environ.get("M3_PORT", "8004")             # ② keeps :8003, M1 :8002
SERVED = os.environ.get("M3_SERVED", "minimax-m3-vq")
MAXLEN = int(os.environ.get("M3_MAXLEN", "40960"))

import m3_quant_vq  # noqa: E402  registers "autoround_mixed_vq" (imports m3_quant too)
import serve_m3_official  # noqa: E402  registrations (loader) + shared config override


def _ov(config):
    """② override (arch force + FORCE dict) + select the VQ quant method."""
    config = serve_m3_official._override_quant_method(config)
    qc = getattr(config, "quantization_config", None)
    if isinstance(qc, dict) and qc.get("quant_method") in ("auto-round", "autoround_mixed"):
        qc["quant_method"] = "autoround_mixed_vq"
    return config


from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402

_cgmode = os.environ.get("M3_CUDAGRAPH_MODE")
_comp_cfg = {"cudagraph_mode": _cgmode} if _cgmode else {}
# VQ codebooks are lazily loaded (torch.load + H2D) on the FIRST MoE forward per
# device; with cudagraph_num_of_warmups=0 that first forward happens INSIDE graph
# capture -> allocation during capture -> illegal memory access. Warmups > 0 run
# eager forwards first, priming the codebook cache before capture.
_cgwarm = os.environ.get("M3_CUDAGRAPH_WARMUPS")
if _cgwarm:
    _comp_cfg["cudagraph_num_of_warmups"] = int(_cgwarm)

_ENGINE_ARGS = AsyncEngineArgs(
    compilation_config=_comp_cfg,
    model=CKPT,
    # NOTE: from_cli_args is replaced with this fixed object, so engine-level
    # settings (incl. the reasoning parser) must be set HERE — the CLI
    # --reasoning-parser below only reaches the OpenAI layer, not the engine.
    reasoning_parser=os.environ.get("M3_REASONING_PARSER", "minimax_m3"),
    quantization="autoround_mixed_vq",
    hf_overrides=_ov,
    trust_remote_code=True,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    enable_expert_parallel=True,
    distributed_executor_backend="mp",
    block_size=128,                                  # mandatory for MSA sparse cache
    attention_backend=os.environ.get("M3_ATTN_BACKEND", "TRITON_ATTN"),
    max_model_len=MAXLEN,
    max_num_batched_tokens=2048,
    gpu_memory_utilization=float(os.environ.get("M3_GPU_UTIL", "0.97")),
    max_num_seqs=int(os.environ.get("M3_MAX_SEQS", "4")),
    enforce_eager=os.environ.get("M3_EAGER", "1") == "1",
    disable_custom_all_reduce=True,                  # Blackwell sm_120
    dtype="bfloat16",
)
AsyncEngineArgs.from_cli_args = classmethod(lambda cls, args: _ENGINE_ARGS)

from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import (  # noqa: E402
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

if __name__ == "__main__":
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([
        CKPT,
        "--served-model-name", SERVED,
        "--host", "0.0.0.0", "--port", PORT,
    ])
    # split <mm:think> into the reasoning field (official M3 parser) -> WebUI
    # renders thinking collapsed; content arrives clean for API consumers.
    # NOTE: must be set on the NESTED namespace config — the OpenAI layer reads
    # args.structured_outputs_config.reasoning_parser, and the flat
    # --reasoning-parser is folded in by from_cli_args, which we replace.
    args.structured_outputs_config.reasoning_parser = os.environ.get(
        "M3_REASONING_PARSER", "minimax_m3")
    validate_parsed_serve_args(args)
    print(f"[m3_vq_api] starting OpenAI server on :{PORT} "
          f"(model={SERVED}, ckpt={CKPT}, maxlen={MAXLEN})", flush=True)
    asyncio.run(run_server(args))
