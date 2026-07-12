# Qwen3.6-35B-A3B on Apple Silicon — VQ experts, custom Metal kernel (model #3)

The first vqmoe adapter that does **not** run on vLLM/CUDA at all: the same trained-codebook
VQ math as models #1/#2, ported to **MLX on Apple Silicon** with a hand-written
`mx.fast.metal_kernel` GEMV. Proof that the quantization pipeline is substrate-independent —
only the last-mile dequant kernel is platform work.

**Artifact**: [aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.6bpw](https://huggingface.co/aquaman164/Qwen3.6-35B-A3B-MLX-VQ-2.6bpw)
(11.53 GB, experts avg 2.38 bpw {2bit: 69, 3bit: 11 tensors} + GPTQ-4bit spine, vision bf16).

| metric | value |
|---|---|
| quality (think-ja PPL vs bf16) | **+12.4%** — vs +17.5% for same-size scalar GPTQ |
| decode / prefill (M-series, 48 GB) | **~66 tok/s / ~213 tok/s** |
| min unified memory | ~16 GB |

## Files

| file | role |
|---|---|
| `vq_switch.py` | `VQSwitchLinear` (packed-code unpack + codebook decode), fused `VQSwitchGLU`, `load_vq_model()` |
| `vq_kernel.py` | Metal kernels: simdgroup-per-row VQ GEMV (`vq_gemv2`) + fused gate·up·SiLU (`vq_swiglu`) |
| `vq_serve.py` | OpenAI-compatible server (mlx_lm.server with the VQ loader patched in) |
| `vq_generate.py` | one-shot generation smoke test |
| `vq_test.py` | Metal smoke + golden check vs CUDA reference + kernel-vs-reference + micro-bench |
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
