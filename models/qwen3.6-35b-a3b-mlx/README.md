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

Env knobs:

| var | default | effect |
|---|---|---|
| `VQ_KERNEL` | `2` | `2`=simdgroup GEMV, `1`=simple kernel, `0`=pure-MLX reference |
| `VQ_FUSED` | `1` | fused gate·up·SiLU dispatch |
| `VQ_PREFILL_N` | `256` | batch size above which prefill uses dequant+`gather_mm` (0 = off) |
| `VQ_PREFILL_STEP` | `2048` | server prefill chunk. Bigger amortizes dequant (8192→438 vs 2048→309 tok/s) but the per-chunk fp16 transient scales with it and can OOM the Metal working set; raise only with headroom to spare |
| `VQ_PREFILL_MIN_HEADROOM_GB` | `16` | fast path falls back to the memory-cheap GEMV kernels when free Metal memory drops below this — prevents a big prompt + grown prompt-cache from OOM-crashing the server |
| `VQ_CACHE` | `1` | persist system-segment KV entries to disk — fresh sessions skip re-prefilling the (byte-stable) system+tools prefix |
| `VQ_CACHE_DIR` / `VQ_CACHE_DISK_GB` | `~/.vq3/prompt_cache` / `8` | persistence location / disk budget |
| `VQ_CACHE_RAM_GB` | `4` | cap on the in-RAM (GPU) KV prompt cache. It grew to ~9.6 GB across cached conversations and, on the 10.5 GB model + a big-context prefill, overflowed a 48 GB Mac's Metal working set. Capping keeps the resident baseline low; raise it on machines with more unified memory |
| `VQ_MTP` | `0` | self-speculative decoding via the bundled MTP head (+8% at temp 0 only; see MODEL_CARD) |

See [MODEL_CARD.md](MODEL_CARD.md) for the full recipe, artifact format, kernel design
and measured ablations.
