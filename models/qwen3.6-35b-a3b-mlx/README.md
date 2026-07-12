# Qwen3.6-35B-A3B on Apple Silicon — VQ experts, custom Metal kernel (model #3)

The first vqmoe adapter that does **not** run on vLLM/CUDA at all: the same trained-codebook
VQ math as models #1/#2, ported to **MLX on Apple Silicon** with a hand-written
`mx.fast.metal_kernel` GEMV. Proof that the quantization pipeline is substrate-independent —
only the last-mile dequant kernel is platform work.

**Artifacts** (same adapter serves both):

| build | size | think-ja PPL vs bf16 | vs scalar at same size |
|---|---|---|---|
| [VQ-2.4bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.4bpw) — experts MCKP {1.5bit: 51, 2bit: 13, 3bit: 16} | 10.49 GB | **+19.5%** | smallest scalar (2.7bpw, 12.0 GB): +17.5% |
| [VQ-2.6bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.6bpw) — experts {2bit: 69, 3bit: 11} | 11.53 GB | **+12.4%** | scalar 2.7bpw: +17.5% |
| [VQ-3.4bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-3.4bpw) — experts uniform 3bit | 15.02 GB | **+6.0%** | scalar 3.5bpw: +6.6% |

VQ dominates scalar at **both** ends of the curve, and the 2.4bpw build (d=8 1.5-bit tier)
reaches a size scalar can't usefully occupy at all. Decode ~66 tok/s (2.6bpw) / prefill
~213 tok/s on an M-series 48 GB; minimum unified memory ~16 GB (2.4/2.6bpw) / ~24 GB (3.4bpw).

## Files

| file | role |
|---|---|
| `vq_switch.py` | `VQSwitchLinear` (packed-code unpack + codebook decode), fused `VQSwitchGLU`, `load_vq_model()` |
| `vq_kernel.py` | Metal kernels: simdgroup-per-row VQ GEMV + fused gate·up·SiLU, d=4 (`vq_gemv2`/`vq_swiglu`) and d=8 (`*_d8`) variants |
| `vq_serve.py` | OpenAI-compatible server (mlx_lm.server with the VQ loader patched in; survives client disconnects mid-stream) |
| `vq_proxy.py` | Anthropic Messages ⇄ OpenAI proxy — lets **Claude Code** use the VQ server as its backend (streaming, tools, system-message folding) |
| `vq_generate.py` | one-shot generation smoke test |
| `vq_test.py` | Metal smoke + golden check vs CUDA reference + kernel-vs-reference + micro-bench |
| `vq_ktest.py` | synthetic all-tier kernel test (no artifact needed): correctness vs reference + bench |
| `run_mac.sh` | launcher |

## Run

```bash
pip install "mlx>=0.31" "mlx-lm>=0.31"
hf download aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.6bpw --local-dir qwen-vq
sh run_mac.sh qwen-vq 8090        # or: python vq_serve.py --model qwen-vq --port 8090
```

Env knobs: `VQ_KERNEL=2|1|0` (simdgroup / simple / pure-MLX reference),
`VQ_FUSED=1|0` (fused swiglu dispatch). Defaults are the fast path; `0/0` is the
kernel-free reference used for verification.

See [MODEL_CARD.md](MODEL_CARD.md) for the full recipe, artifact format, kernel design
and measured ablations.
