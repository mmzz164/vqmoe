# Qwen3.6-35B-A3B MLX-VQ — technical card (vqmoe model #3)

Trained-codebook VQ (AQLM-style) quantization of Qwen3.6-35B-A3B (35B MoE VLM: 40 layers,
256 experts / top-8, hybrid linear-attention, `qwen3_5_moe`), served on **Apple Silicon via MLX**
with custom Metal kernels. Quantized on 2×RTX PRO 6000 (CUDA); runs on any ≥16 GB M-series Mac.

## Why VQ here

Scalar affine quantization collapses below ~3 bpw on this model (measured, GPTQ-compensated,
think-ja holdout PPL vs bf16):

| experts avg bpw | scalar GPTQ | **VQ + GPTQ** |
|---|---|---|
| ~3.2 (15.4 GB total) | +6.6% | — |
| **~3.1 (15.0 GB total, uniform 3bit)** | — | **+6.0%** |
| ~2.6 (12.0 GB total) | +17.5% | — |
| **~2.4 (11.5 GB total, {2bit:69, 3bit:11})** | — | **+12.4%** |
| **~2.0 (10.45 GB total, {1.5bit:51, 2bit:13, 3bit:16})** | — | **+19.5%** |
| plain-RTN reference @2.6 | +47% | — |

VQ is smaller *and* ~5 pts better than the scalar build — the codebook captures subvector
structure a 4-point scalar grid cannot. (An AWQ pass on top of scalar GPTQ was also measured
and **rejected**: +20.6%, worse than GPTQ alone — its per-channel scales widen the intra-group
range that a 2-bit grid must cover.)

## Recipe (quantization side, CUDA)

1. **Fisher/gxw allocation** — empirical per-expert Fisher (gbar/xbar) from a ja-think calibration
   corpus; MCKP over a menu per grouped expert tensor. 2.6bpw build: (bits × group-size) scalar
   menu mapped to VQ tiers, chosen {2bit: 69, 3bit: 11}; the 3-bit budget lands on late layers
   (L33-39). 2.4bpw build: **VQ-native tier menu {1.5, 2, 3}** costed by plain nearest-codeword
   encode (same proxy role as RTN), chosen {1.5bit: 51, 2bit: 13, 3bit: 16} at experts avg
   2.0 bpw eff — a barbell: cheap tensors drop to 1.5-bit, sensitive late layers keep 3-bit.
2. **Qwen-native codebooks** — k-means on std-normalized subvectors (GS=128 groups).
   Transfer test showed GLM-fit production books lose 0.6–6.1% reconstruction RMS on Qwen,
   so refit: d=4 K=256 (2-bit), d=4 K=4096 (3-bit), and a **d=8 K=4096 book for the 1.5-bit
   tier** (12 bits / 8 weights; fit rms 0.401).
3. **GPTQ-VQ encode** — column-block GPTQ sweep with nearest-codeword quantizer
   (`gptqvq_encode`), pooled per-layer input Hessians (16×2048 ja tokens), batched over
   [E·R, C] rows: all 256 experts of a tensor encode in one call.
4. **Spine** — GPTQ 4bit gs64 (linear-attn qkv/z/out ×30, attn q/k/v/o ×10, shared experts ×40,
   with their own input Hessians), lm_head 6bit, router 8bit, embeddings 4bit, vision bf16.
   Assembled by cloning the scalar 2.7bpw MLX build and swapping `switch_mlp` tensors.

## Artifact format

```
language_model.model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.vq_codes   int32 [E, R, PC]
language_model.model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.vq_scales  fp16  [E, R, C/128]
vq_codebooks.safetensors                                          cb<tier> [K, d] fp16 (used tiers only)
config.json["vq"] = {modules: {path: {vq_bits, d, K, nbits, in_dims, norm_group}}, ...}
```
Codes are LSB-first bit-packed indices (8-bit for K=256, 12-bit for K=4096) in uint32 words.
Weight reconstruction: `W[r, c] = cb[idx(r, c/d)][c%d] * scale[r, c/128]` (d = 4, or 8 for the
1.5-bit tier).
The scalar-quantized spine keeps stock mlx-lm `quantization` config entries; `switch_mlp`
entries are removed so the stock loader never touches VQ modules.

## Serving side (MLX / Metal)

- `VQSwitchLinear` — drop-in for mlx_lm's `QuantizedSwitchLinear` (same `(x, indices,
  sorted_indices)` contract). Pure-MLX reference path (`VQ_KERNEL=0`) + Metal kernels.
- **Kernels** (`mx.fast.metal_kernel`, JIT from Python, no build step):
  - `vq_gemv2` (d=4) / `vq_gemv2_d8` (d=8): one simdgroup (32 lanes) per output row; lanes
    stride subvectors with half4 codebook/activation loads; `simd_sum` reduction. 8-bit codes
    are plain byte reads; 12-bit codes unpack with a two-word straddle read. The d=8 variant
    is *faster* per dispatch than d=4/12-bit (half the subvectors → half the index unpacks).
  - `vq_swiglu` / `vq_swiglu_d8`: fused silu(gate(x))·up(x) — both projections + activation in
    **one dispatch** per layer (gate/up share geometry and tier by construction; down may be a
    different tier and runs its own GEMV).
- `VQSwitchGLU` — replaces the whole SwitchGLU: one fused dispatch + one down dispatch,
  reusing mlx_lm's `_gather_sort/_scatter_unsort` for the many-token path. The fused kernel is
  selected by the gate/up module's `d` at call time.
- `vq_serve.py` — patches `mlx_lm.server`'s loader: any model dir whose config has a `"vq"`
  section loads through `load_vq_model`; everything else falls back to stock mlx_lm.

## Measured (M-series 48 GB, macOS 26.5)

| stage | decode tok/s |
|---|---|
| pure-MLX reference (`VQ_KERNEL=0`) | 4.2 |
| Metal GEMV v1 (thread-per-row) | 46 |
| simdgroup + half4 (v2) | 61 |
| + fused swiglu (`VQ_FUSED=1`) | **66** |
| null-VQ ablation (framework+spine ceiling) | 87 |

(kernel-progression table measured on the 2.6bpw build.) Per build, kernels + fused:
**2.4bpw 73 tok/s** (fewest expert bytes + the d=8 tier's lighter unpack), **2.6bpw 66**,
**3.4bpw 52** (all-12-bit, most bytes). Prefill ~213 tok/s. The remaining gap to 87 is stock
mlx_lm spine work (A3B: expert bytes are only ~30% of decode traffic). Outputs are bit-identical
between fused/unfused and match the CUDA fakequant reference (golden test, rel ≤7e-4) — so the
published quality numbers are the served model's numbers.

## Portability notes / traps

- Qwen3.6's RMSNorm is Gemma-style `out = x̂ · (1 + w)` — any weight-space reparametrization
  must use `w' = (1+w)/s − 1`, not `w/s`.
- With mixed subvector widths in one layer stack (d=4 and d=8 tiers), every kernel-dispatch
  guard must key on the *module's* `d` — a d=8 pair silently entering the d=4 fused kernel
  reads misaligned codebook entries and produces garbage, not an error.
- Claude Code (Agent SDK) injects extra `system`-role messages *inside* `messages[]`; Qwen chat
  templates `raise_exception` unless the single system message is first. `vq_proxy.py` folds
  non-leading system messages into user turns. (The upstream error surfaces in Claude Code as a
  misleading "model may not exist".)
- A streaming client that disconnects mid-generation (Claude Code Esc) raises BrokenPipe inside
  mlx_lm.server's generation write path and can take the whole server down — `vq_serve.py` wraps
  `handle_completion` to drop that request and keep serving.
- **Stock mlx-lm sanitize treats any `mtp.*` key as "raw HF checkpoint"** and +1-shifts every
  backbone norm — shipping MTP weights inside an already-converted artifact poisons a stock-path
  load into multilingual garbage. The loader strips mtp keys before sanitize when MTP is off.
- Qwen3.6's MTP norm weights straddle the mean≈0.5 boundary (pre_fc norms land at 0.27/0.49
  post-shift), so magnitude heuristics for raw-vs-shifted detection misfire in both directions;
  we ship final +1-shifted values and shield them from sanitize instead.

## Prefill fast path + persistent prompt cache

Decode-oriented GEMV kernels are the wrong shape for prefill: every (token, row)
pair re-decodes its codebook entries, so a 47k-token Claude Code system prompt took
~3 minutes (256 tok/s). Two fixes:

1. **Dequant+GEMM prefill** (`VQ_PREFILL_N`): above a batch threshold each expert
   tensor is dequantized once per chunk and dispatched through `mx.gather_mm`
   (mlx-lm's own SwitchLinear pattern). The dequant repeats per prefill chunk, so
   bigger chunks amortize it: 256 → 309 tok/s at the stock 2048-token step,
   **438 tok/s** at 8192 (now the server default via `VQ_PREFILL_STEP`). Decode is
   untouched (still the GEMV kernels). Outputs match level-0 to ≤6e-4.
2. **Persistent prompt cache** (`VQ_CACHE`): mlx-lm's server already snapshots
   prompt-cache entries at chat *segment* boundaries — including one at the end of
   the system segment, which is byte-stable across Claude Code sessions.
   `vq_serve.py` subclasses `LRUPromptCache` to write those system entries to disk
   (background thread) and preload them at startup, so a fresh server start skips
   re-prefilling the system prefix entirely. Trap: worker threads need their own
   MLX stream (`mx.set_default_stream(mx.new_stream(...))`) or every array op
   raises `There is no Stream(gpu, 0) in current thread`.

**Prefill memory vs the Metal limit.** The dequant+GEMM fast path materializes fp16
expert weights plus big-batch activations — a multi-GB transient that scales with
`VQ_PREFILL_STEP`. On a 48 GB Mac (Metal working set ≈ 40 GB) an 8192-step prefill of a
27k-token prompt spiked >21 GB and, on top of a prompt cache that had grown to ~8 GB,
overflowed the working set. A Metal command-buffer OOM is an **uncatchable C++ abort**
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`) — it kills the whole server, not just the
request, so it must be *prevented*. Two guards: the shipped `VQ_PREFILL_STEP` default is
`2048` (transient ~5 GB), and `_use_prefill()` checks live headroom
(`mx.get_active_memory()` vs `max_recommended_working_set_size`) before every fast-path
dispatch, falling back to the memory-cheap GEMV kernels when free memory drops below
`VQ_PREFILL_MIN_HEADROOM_GB`. So a big prompt on a loaded machine degrades to slow-but-safe
instead of crashing.

## MTP self-speculative decoding (opt-in)

The 2.4bpw artifact ships the checkpoint's MTP head (0.845B params quantized to ~0.5 GB:
experts 4bit gs64 in switch_mlp form, attention/shared 8bit, fc/router/norms fp16). With
[oMLX](https://github.com/jundot/omlx)'s mlx-lm PR#990 patches importable, `VQ_MTP=1` enables
the draft/verify cycle (2-token verify with `n_confirmed=1` + one MTP forward per cycle).

Measured (M-series 48 GB, 300-token generations): accept 82% (greedy) / 87% (temp 1),
tokens/cycle 1.83-1.87 — but a top-8 MoE pays ~2x expert bytes on the 2-token verify
(21.0ms vs 13.6ms single-step), so the net is **+8% at temperature 0** (bit-identical
outputs vs non-MTP, verified) and **-17% at temperature 1** (sampled path adds per-cycle
softmax/sampling). Off by default; useful for greedy workloads only. This is the honest
MoE-at-batch-1 speculative-decoding economics — dense models are where MTP shines.
- vLLM workers rename themselves `VLLM::Worker…` — `pkill -f` by script name misses them.
- `mx.view(int32 → uint32)` before shifts: MLX right-shift is arithmetic on signed dtypes.
- safetensors `framework="numpy"` cannot read bf16; route through torch or pre-dump f32.
- The whole kernel loop was developed **over SSH** (no keychain unlock needed —
  `mx.fast.metal_kernel` is runtime shader compilation, not codesigning).
