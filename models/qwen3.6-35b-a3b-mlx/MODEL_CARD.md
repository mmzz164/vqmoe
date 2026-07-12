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
- vLLM workers rename themselves `VLLM::Worker…` — `pkill -f` by script name misses them.
- `mx.view(int32 → uint32)` before shifts: MLX right-shift is arithmetic on signed dtypes.
- safetensors `framework="numpy"` cannot read bf16; route through torch or pre-dump f32.
- The whole kernel loop was developed **over SSH** (no keychain unlock needed —
  `mx.fast.metal_kernel` is runtime shader compilation, not codesigning).
