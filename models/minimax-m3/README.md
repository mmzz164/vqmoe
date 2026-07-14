# models/minimax-m3 — VQ 2.4 bpw adapter

Serve the [MiniMax-M3 VQ 2.4-bit checkpoint](https://huggingface.co/aquaman164/MiniMax-M3-VQ-2.4bit)
on 2× RTX PRO 6000 (sm_120). See [`MODEL_CARD.md`](MODEL_CARD.md) for what the build is and its
quality; this file is how to run it.

## Why M3 is different from the GLM adapter

M3 serves on **official vLLM's native MiniMax-M3 model** (vLLM 0.23.1 nightly), which already ships
the MSA lightning indexer. So:

- **The `vqmoe/vllm-sm120/` sparse patch is NOT used** — M3's native indexer runs on the stock
  `TRITON_ATTN` backend on sm_120. This is a *cleaner* reproducibility story than GLM (no hand-built
  sparse fork): official vLLM + OneCompression VQ kernels + the small adapter here is the whole
  stack.
- The adapter's job is only **checkpoint-key translation and quantization dispatch**: our checkpoint
  is exported with transformers-style names, the official vLLM model expects MiniMax-native names,
  and two fused linears (qkv+indexer, gate_up) are de-quantized to bf16 at load. All of that lives
  in the files below; the routed experts (the whole point of the quantization) load and run fully
  quantized through the stock FusedMoE path.

## Files (all copied verbatim from the deployment)

| file | role |
|---|---|
| `m3_vq_api_server.py` | **entrypoint** — OpenAI API server (:8004). Sets engine args + the reasoning parser. |
| `m3_quant_vq.py` | registers `autoround_mixed_vq` — the VQ-aware quant config (keeps the per-expert `format:"vq"` marker so dispatch reaches the VQ kernels). |
| `m3_quant.py` | registers `autoround_mixed` — the scalar base config the VQ one extends. |
| `serve_m3_official.py` | the config override (architecture force + the FORCE dict incl. `n_shared_experts=1`) shared with the long-context ② port. |
| `m3_official_loader.py` | the key-translating CausalLM loader (transformers→MiniMax names; selective de-quant of the two fused linears). |

Import graph at launch: `m3_vq_api_server` → `m3_quant_vq` (→ `m3_quant`) + `serve_m3_official`
(→ `m3_official_loader`). All five must sit in one importable directory.

## Paths you must edit

These files carry machine-local absolute paths (a faithful record of the deployment, not a
turn-key installer). Before running elsewhere, edit:

1. **OneCompression checkout** — every `sys.path.insert(0, "/home/mizugaihiros01/work/onecomp")`
   → your clone of [OneCompression](https://github.com/mmzz164/OneCompression).
2. **This directory** — every `sys.path.insert(0, "/var/hf/vllm_m3")` → the absolute path of
   `models/minimax-m3/` (so the five files import each other).
3. **Checkpoint** — `M3VQ_CKPT` (in `model_spec.sh`) / the `CKPT` default in `m3_vq_api_server.py`
   → where you unpacked the HF download.
4. **VQ codebooks** — `m3_quant_vq.py` / the VQ kernel loads codebooks from `VQ_CODEBOOKS_DIR`
   (default `/var/hf/glm_quant`). Point it at the directory holding
   `vq_codebook_{d8_k256_m1,d4_better,d4_k4096_m1}.pt` (shipped with the OneCompression VQ tooling;
   these GLM-fit codebooks transfer to M3 within ±0.5% reconstruction RMS).

The `__main__` self-test paths in `m3_quant.py` / `m3_quant_vq.py` point at a `quantization_config.json`
and are only for the offline dispatch self-test, not for serving.

## Run

```bash
# 1. install the pinned env (see repo root README "Dependencies"): official vLLM 0.23.1 nightly
#    with native MiniMax-M3, transformers 5.12, and the OneCompression VQ kernels on PYTHONPATH.
# 2. edit the paths above.
# 3. review knobs:
source model_spec.sh          # documents/export the env defaults
# 4. serve:
python m3_vq_api_server.py     # OpenAI API on :8004, model "minimax-m3-vq"
```

Smoke test:
```bash
curl -s localhost:8004/v1/models
curl -s localhost:8004/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3-vq","messages":[{"role":"user","content":"日本の首都は?"}],"max_tokens":30}'
```

## Multi-GPU / other architectures

The server is parameterized so the same file runs on different GPU counts and architectures — the
~130 GiB weights need enough GPUs to hold the TP shards plus KV, so `M3_TP` **must** be tunable.
Because it rides on **official** vLLM's native MiniMax-M3, nothing here is Blackwell-specific; only
one default (`M3_EAGER=1`, dodging an sm_120 CUDA-graph capture race) reflects our hardware.

Reference configs:

| hardware | launch env | throughput |
|---|---|---|
| 2× RTX PRO 6000 (sm_120, 96 GB) | `M3_TP=2 M3_EAGER=1` (defaults) | ~7 tok/s (eager — CUDA graphs race here) |
| 8× RTX 3090 (sm_86, 24 GB) | `M3_TP=8 M3_EAGER=0 M3_CUDAGRAPH_WARMUPS=2 M3_MAX_BATCHED_TOKENS=4096` | ~20 tok/s (community-reported) |

Two things the 8×3090 run confirmed: (1) the model runs fine on **Ampere** — no Blackwell dependency;
(2) **CUDA graphs work off-sm_120** (`M3_EAGER=0` + warmups), for ~2–3× decode. Our eager-only default
is purely an sm_120 capture-race workaround, not a model limitation. Knobs: `M3_TP`, `M3_PP`,
`M3_EXPERT_PARALLEL`, `M3_MAX_BATCHED_TOKENS`, `M3_DISABLE_CUSTOM_AR` (see `model_spec.sh`). `block_size`
stays 128 (mandatory for the MSA sparse cache — a correctness requirement, not a knob).

## Reproducibility status

Cleaner than the GLM adapter — no hand-compiled sparse fork. What's needed for a same-as-ours run:

| piece | status |
|---|---|
| Adapter (this dir: server, VQ config, key-translation loader) | ✅ here |
| VQ dequant kernels + codebooks | ✅ public (OneCompression) |
| **Base vLLM** | ✅ **official** 0.23.1 nightly (native MiniMax-M3) — no fork, no patch |
| Quantized checkpoint | ✅ [HF](https://huggingface.co/aquaman164/MiniMax-M3-VQ-2.4bit) |
| Machine-local paths | ⚠ edit the four above |
| One-shot installer / pinned lockfile | ⏳ not provided; versions in the repo-root README |
