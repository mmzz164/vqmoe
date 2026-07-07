# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for the GLM-5.2 AutoBit fit (mixed 1/2/3-bit
experts + asym-MSE spine) on native vLLM GlmMoeDsaForCausalLM (dense MLA on
sm_120), for use with OpenWebUI etc.

Reuses serve_glm.py's registrations (glm_quant: autoround_mixed + the
GLM_FORCE_DENSE patch) and its _override_quant_method. Same AsyncEngineArgs as
serve_glm.py's LLM kwargs; we replace the OpenAI server's CLI-parsed engine args
with ours (m3_official_api_server.py trick).

Launch via start_glm_api.sh (sets CUDA_HOME/LD_LIBRARY_PATH/PYTHONPATH/pylibs).
Env: GLM_PORT(8001), GLM_SERVED(glm-5.2), GLM_MAXLEN(2048), GLM_GPU_UTIL(0.982),
     GLM_KV_BLOCKS(20), GLM_EAGER(0=cudagraphs), GLM_MAX_SEQS(1), GLM_MAXBATCH(8),
     plus the model knobs GLM_ASYM_SPINE / GLM_FORCE_DENSE / MIXED_MOE_GROUPED.
Test: curl http://localhost:8001/v1/models
"""
import os
import sys
import asyncio

os.environ.setdefault("MIXED_MOE_GROUPED", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, "/var/hf/glm_serve")
sys.path.insert(0, "/var/hf/glm_quant")  # for lz_penalty_logitproc (the loop-fix)

# Importing serve_glm runs the registrations (glm_quant -> autoround_mixed +
# the GLM_FORCE_DENSE deepseek_v2.hasattr patch, both env-gated) and defines
# _override_quant_method. Its LLM build is under __main__, so import is safe.
import serve_glm  # noqa: E402
_ov = serve_glm._override_quant_method

CKPT = serve_glm.CKPT
PORT = os.environ.get("GLM_PORT", "8001")
SERVED = os.environ.get("GLM_SERVED", "glm-5.2")
MAXLEN = int(os.environ.get("GLM_MAXLEN", "2048"))
# Default to the NO-THINK template: GLM-5.2's reasoning_effort=max thinking does
# not terminate in a usable token budget for casual chat, so default to direct
# answers. (Pass GLM_CHAT_TEMPLATE to override, or chat_template_kwargs
# enable_thinking=true per request to re-enable reasoning.)
CHAT_TEMPLATE = os.environ.get(
    "GLM_CHAT_TEMPLATE", "/var/hf/glm_serve/chat_template_nothink.jinja")

from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
# Loop-fix: LZ anti-repetition + </think> budget forcing (round-5's validated fix, finally wired
# into production). Both are env-gated (GLM_LZ_PENALTY / GLM_BUDGET) and no-op when off, so the
# default/scalar behaviour is unchanged. They only steer sampling; weights/vLLM untouched.
from lz_penalty_logitproc import LZPenaltyLogitsProcessor, BudgetForceLogitsProcessor  # noqa: E402

_kw = dict(
    model=CKPT,
    quantization="autoround_mixed",
    hf_overrides=_ov,
    trust_remote_code=True,
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    enable_expert_parallel=True,
    distributed_executor_backend="mp",
    block_size=128,
    max_model_len=MAXLEN,
    max_num_batched_tokens=int(os.environ.get("GLM_MAXBATCH", "8")),
    gpu_memory_utilization=float(os.environ.get("GLM_GPU_UTIL", "0.982")),
    max_num_seqs=int(os.environ.get("GLM_MAX_SEQS", "1")),
    enforce_eager=os.environ.get("GLM_EAGER", "0") == "1",
    disable_custom_all_reduce=True,
    dtype="bfloat16",
    # Loop-fix processors (env-gated: GLM_LZ_PENALTY=1 enables LZ anti-loop; GLM_BUDGET>0 forces
    # </think> after that many thinking tokens). Always attached; no-op when their env is off.
    logits_processors=[LZPenaltyLogitsProcessor, BudgetForceLogitsProcessor],
    # NOTE: the OLD broken 2bpw model needed aggressive anti-repetition (rep 1.3 /
    # freq 0.7) to avoid loops. The multilingual re-quant fixed the degeneration, and
    # those heavy penalties now CAUSE Japanese->Chinese drift: they suppress repeated
    # JA particles (は/の/です), making same-script Chinese tokens relatively likelier,
    # so a JA answer slides into Chinese mid-sentence. Verified: rep 1.3/freq 0.7 drifts,
    # rep 1.05/freq 0 stays clean Japanese. Light rep still discourages loops.
    override_generation_config={
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "frequency_penalty": 0.0,
    },
)
_kvb = os.environ.get("GLM_KV_BLOCKS", "")
if _kvb:
    _kw["num_gpu_blocks_override"] = int(_kvb)
_ENGINE_ARGS = AsyncEngineArgs(**_kw)
# Make the OpenAI server use OUR engine args (the CLI path mis-resolves config).
AsyncEngineArgs.from_cli_args = classmethod(lambda cls, args: _ENGINE_ARGS)

from vllm.entrypoints.openai.api_server import run_server  # noqa: E402
from vllm.entrypoints.openai.cli_args import (  # noqa: E402
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

if __name__ == "__main__":
    cli = [CKPT, "--served-model-name", SERVED, "--host", "0.0.0.0", "--port", PORT]
    if CHAT_TEMPLATE:
        cli += ["--chat-template", CHAT_TEMPLATE]
    # GLM-5.2 is a reasoning model (emits CoT then </think> then the answer).
    # The glm47 parser splits that into reasoning_content vs content so OpenWebUI
    # shows a collapsible "thinking" section and a clean answer.
    _rp = os.environ.get("GLM_REASONING_PARSER", "glm47")
    if _rp and _rp.lower() != "none":
        cli += ["--reasoning-parser", _rp]
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(cli)
    # The CLI --reasoning-parser lands in args.reasoning_parser, but the OpenAI chat
    # serving reads args.structured_outputs_config.reasoning_parser (api_server.py:506).
    # vLLM normally syncs the two inside from_cli_args -> but we monkeypatch from_cli_args
    # to a FIXED _ENGINE_ARGS (to inject our engine config), which skips that sync. Without
    # this line the reasoning parser never engages: the whole CoT (<think>...</think>) leaks
    # into `content` and reasoning_content stays null.
    if _rp and _rp.lower() != "none":
        args.structured_outputs_config.reasoning_parser = _rp
    validate_parsed_serve_args(args)
    print(f"[glm_api] starting OpenAI server on :{PORT} (model={SERVED}, "
          f"maxlen={MAXLEN}, eager={_kw['enforce_eager']}, "
          f"chat_template={'yes' if CHAT_TEMPLATE else 'no'})", flush=True)
    asyncio.run(run_server(args))
