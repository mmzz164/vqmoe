"""Fused VQ gather-GEMV Metal kernel (v3 level-1) for map12 (d=4, nbits 8|12).

One thread per (pair n, output row r):
  y[n,r] = sum_g scale[e,r,g] * sum_{sub in g} dot(cb[idx(e,r,sub)], x[n, sub*4 .. +4])
codes are LSB-first packed (8-bit idx for K256 / 12-bit for K4096) in uint32 words.
Correctness reference = VQSwitchLinear._decode (level-0); see vq_test kernel section.
"""
import mlx.core as mx

_SRC = """
    uint r = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    const uint C     = (uint)params[0];
    const uint R     = (uint)params[1];
    const uint nbits = (uint)params[3];
    const uint PC    = (uint)params[4];
    const uint NG    = (uint)params[5];
    if (r >= R) return;
    uint e = (uint)eidx[n];
    device const uint* crow  = codes  + ((ulong)e * R + r) * PC;
    device const half* srow  = scales + ((ulong)e * R + r) * NG;
    device const half* xr    = x + (ulong)n * C;
    float acc = 0.0f;
    const uint SPG = 32u;                       // subvectors (d=4) per 128-group
    for (uint g = 0; g < NG; ++g) {
        float s = (float)srow[g];
        float gacc = 0.0f;
        uint base = g * SPG;
        for (uint j = 0; j < SPG; ++j) {
            uint k = base + j;
            uint idx;
            if (nbits == 8u) {
                idx = (crow[k >> 2] >> ((k & 3u) * 8u)) & 255u;
            } else {
                uint pos = k * 12u; uint w = pos >> 5; uint off = pos & 31u;
                uint v = crow[w] >> off;
                if (off > 20u) { v |= crow[w + 1u] << (32u - off); }
                idx = v & 4095u;
            }
            device const half* cv = cb + (ulong)idx * 4u;
            uint c0 = k * 4u;
            gacc += (float)cv[0] * (float)xr[c0]
                  + (float)cv[1] * (float)xr[c0 + 1u]
                  + (float)cv[2] * (float)xr[c0 + 2u]
                  + (float)cv[3] * (float)xr[c0 + 3u];
        }
        acc += s * gacc;
    }
    y[(ulong)n * R + r] = (half)acc;
"""

_KERNEL = mx.fast.metal_kernel(
    name="vq_gemv_d4",
    input_names=["x", "codes", "scales", "cb", "eidx", "params"],
    output_names=["y"],
    source=_SRC,
)

# ---- v2: one simdgroup (32 lanes) per output row, half4 vector loads ----
_SRC2 = """
    uint tid  = thread_position_in_grid.x;
    uint r    = tid / 32u;
    uint lane = tid % 32u;
    uint n    = thread_position_in_grid.y;
    const uint C     = (uint)params[0];
    const uint R     = (uint)params[1];
    const uint nbits = (uint)params[3];
    const uint PC    = (uint)params[4];
    if (r >= R) return;
    uint e = (uint)eidx[n];
    device const uint*  crow = codes + ((ulong)e * R + r) * PC;
    device const half*  srow = scales + ((ulong)e * R + r) * (C >> 7);
    device const half4* xr4  = (device const half4*)(x + (ulong)n * C);
    device const half4* cb4  = (device const half4*)cb;
    const uint nsub = C >> 2;
    float acc = 0.0f;
    for (uint k = lane; k < nsub; k += 32u) {
        uint idx;
        if (nbits == 8u) {
            idx = (crow[k >> 2] >> ((k & 3u) * 8u)) & 255u;
        } else {
            uint pos = k * 12u; uint w = pos >> 5; uint off = pos & 31u;
            uint v = crow[w] >> off;
            if (off > 20u) { v |= crow[w + 1u] << (32u - off); }
            idx = v & 4095u;
        }
        half4 cv = cb4[idx];
        half4 xv = xr4[k];
        float s  = (float)srow[k >> 5];
        acc += s * ((float)cv.x * (float)xv.x + (float)cv.y * (float)xv.y
                  + (float)cv.z * (float)xv.z + (float)cv.w * (float)xv.w);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[(ulong)n * R + r] = (half)acc; }
"""

_KERNEL2 = mx.fast.metal_kernel(
    name="vq_gemv_d4_sg",
    input_names=["x", "codes", "scales", "cb", "eidx", "params"],
    output_names=["y"],
    source=_SRC2,
)


def vq_gemv2(x_f16, codes_u32, scales_f16, cb_f16, eidx_i32, nbits):
    """simdgroup-per-row variant. Same contract as vq_gemv."""
    N, C = x_f16.shape
    E, R, PC = codes_u32.shape
    NG = scales_f16.shape[-1]
    params = mx.array([C, R, 4, nbits, PC, NG], dtype=mx.int32)
    (y,) = _KERNEL2(
        inputs=[x_f16, codes_u32, scales_f16, cb_f16, eidx_i32, params],
        output_shapes=[(N, R)],
        output_dtypes=[mx.float16],
        grid=(R * 32, N, 1),
        threadgroup=(256, 1, 1),
    )
    return y


# ---- v3: fused gate+up GEMV + SiLU (one dispatch per layer instead of two + eltwise) ----
_SRC3 = """
    uint tid  = thread_position_in_grid.x;
    uint r    = tid / 32u;
    uint lane = tid % 32u;
    uint n    = thread_position_in_grid.y;
    const uint C     = (uint)params[0];
    const uint R     = (uint)params[1];
    const uint nbits = (uint)params[3];
    const uint PC    = (uint)params[4];
    if (r >= R) return;
    uint e = (uint)eidx[n];
    ulong row = (ulong)e * R + r;
    device const uint*  cg   = codes_g  + row * PC;
    device const uint*  cu   = codes_u  + row * PC;
    device const half*  sg   = scales_g + row * (C >> 7);
    device const half*  su   = scales_u + row * (C >> 7);
    device const half4* xr4  = (device const half4*)(x + (ulong)n * C);
    device const half4* cb4  = (device const half4*)cb;
    const uint nsub = C >> 2;
    float ag = 0.0f, au = 0.0f;
    for (uint k = lane; k < nsub; k += 32u) {
        uint ig, iu;
        if (nbits == 8u) {
            ig = (cg[k >> 2] >> ((k & 3u) * 8u)) & 255u;
            iu = (cu[k >> 2] >> ((k & 3u) * 8u)) & 255u;
        } else {
            uint pos = k * 12u; uint w = pos >> 5; uint off = pos & 31u;
            uint vg = cg[w] >> off; uint vu = cu[w] >> off;
            if (off > 20u) { vg |= cg[w + 1u] << (32u - off); vu |= cu[w + 1u] << (32u - off); }
            ig = vg & 4095u; iu = vu & 4095u;
        }
        half4 xv = xr4[k];
        half4 g4 = cb4[ig];
        half4 u4 = cb4[iu];
        float dx = (float)g4.x * (float)xv.x + (float)g4.y * (float)xv.y
                 + (float)g4.z * (float)xv.z + (float)g4.w * (float)xv.w;
        float du = (float)u4.x * (float)xv.x + (float)u4.y * (float)xv.y
                 + (float)u4.z * (float)xv.z + (float)u4.w * (float)xv.w;
        ag += (float)sg[k >> 5] * dx;
        au += (float)su[k >> 5] * du;
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0u) {
        float act = ag / (1.0f + metal::exp(-ag));    // silu(gate)
        y[(ulong)n * R + r] = (half)(act * au);
    }
"""

_KERNEL3 = mx.fast.metal_kernel(
    name="vq_swiglu_d4",
    input_names=["x", "codes_g", "scales_g", "codes_u", "scales_u", "cb", "eidx", "params"],
    output_names=["y"],
    source=_SRC3,
)


def vq_swiglu(x_f16, cg_u32, sg_f16, cu_u32, su_f16, cb_f16, eidx_i32, nbits):
    """Fused silu(gate(x)) * up(x) over VQ experts -> intermediate [N, R] f16."""
    N, C = x_f16.shape
    E, R, PC = cg_u32.shape
    params = mx.array([C, R, 4, nbits, PC, C // 128], dtype=mx.int32)
    (y,) = _KERNEL3(
        inputs=[x_f16, cg_u32, sg_f16, cu_u32, su_f16, cb_f16, eidx_i32, params],
        output_shapes=[(N, R)],
        output_dtypes=[mx.float16],
        grid=(R * 32, N, 1),
        threadgroup=(256, 1, 1),
    )
    return y


def vq_gemv(x_f16, codes_u32, scales_f16, cb_f16, eidx_i32, nbits):
    """x [N,C] f16, codes [E,R,PC] uint32, scales [E,R,NG] f16, cb [K,4] f16, eidx [N] int32 -> y [N,R] f16."""
    N, C = x_f16.shape
    E, R, PC = codes_u32.shape
    NG = scales_f16.shape[-1]
    params = mx.array([C, R, 4, nbits, PC, NG], dtype=mx.int32)
    (y,) = _KERNEL(
        inputs=[x_f16, codes_u32, scales_f16, cb_f16, eidx_i32, params],
        output_shapes=[(N, R)],
        output_dtypes=[mx.float16],
        grid=(R, N, 1),
        threadgroup=(min(R, 256), 1, 1),
    )
    return y


# ---- d=8 variants (tier 1.5: K4096, 12-bit idx): simdgroup-per-row, 2x half4 per subvector ----
_SRC2_D8 = """
    uint tid  = thread_position_in_grid.x;
    uint r    = tid / 32u;
    uint lane = tid % 32u;
    uint n    = thread_position_in_grid.y;
    const uint C     = (uint)params[0];
    const uint R     = (uint)params[1];
    const uint nbits = (uint)params[3];
    const uint PC    = (uint)params[4];
    if (r >= R) return;
    uint e = (uint)eidx[n];
    device const uint*  crow = codes + ((ulong)e * R + r) * PC;
    device const half*  srow = scales + ((ulong)e * R + r) * (C >> 7);
    device const half4* xr4  = (device const half4*)(x + (ulong)n * C);
    device const half4* cb4  = (device const half4*)cb;
    const uint nsub = C >> 3;
    float acc = 0.0f;
    for (uint k = lane; k < nsub; k += 32u) {
        uint idx;
        if (nbits == 8u) {
            idx = (crow[k >> 2] >> ((k & 3u) * 8u)) & 255u;
        } else {
            uint pos = k * 12u; uint w = pos >> 5; uint off = pos & 31u;
            uint v = crow[w] >> off;
            if (off > 20u) { v |= crow[w + 1u] << (32u - off); }
            idx = v & 4095u;
        }
        half4 c0 = cb4[idx * 2u];
        half4 c1 = cb4[idx * 2u + 1u];
        half4 x0 = xr4[k * 2u];
        half4 x1 = xr4[k * 2u + 1u];
        float s  = (float)srow[k >> 4];
        acc += s * ((float)c0.x * (float)x0.x + (float)c0.y * (float)x0.y
                  + (float)c0.z * (float)x0.z + (float)c0.w * (float)x0.w
                  + (float)c1.x * (float)x1.x + (float)c1.y * (float)x1.y
                  + (float)c1.z * (float)x1.z + (float)c1.w * (float)x1.w);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[(ulong)n * R + r] = (half)acc; }
"""

_KERNEL2_D8 = mx.fast.metal_kernel(
    name="vq_gemv_d8_sg",
    input_names=["x", "codes", "scales", "cb", "eidx", "params"],
    output_names=["y"],
    source=_SRC2_D8,
)


def vq_gemv2_d8(x_f16, codes_u32, scales_f16, cb_f16, eidx_i32, nbits):
    """d=8 simdgroup-per-row GEMV. cb [K,8] f16; same contract as vq_gemv2."""
    N, C = x_f16.shape
    E, R, PC = codes_u32.shape
    NG = scales_f16.shape[-1]
    params = mx.array([C, R, 8, nbits, PC, NG], dtype=mx.int32)
    (y,) = _KERNEL2_D8(
        inputs=[x_f16, codes_u32, scales_f16, cb_f16, eidx_i32, params],
        output_shapes=[(N, R)],
        output_dtypes=[mx.float16],
        grid=(R * 32, N, 1),
        threadgroup=(256, 1, 1),
    )
    return y


_SRC3_D8 = """
    uint tid  = thread_position_in_grid.x;
    uint r    = tid / 32u;
    uint lane = tid % 32u;
    uint n    = thread_position_in_grid.y;
    const uint C     = (uint)params[0];
    const uint R     = (uint)params[1];
    const uint nbits = (uint)params[3];
    const uint PC    = (uint)params[4];
    if (r >= R) return;
    uint e = (uint)eidx[n];
    ulong row = (ulong)e * R + r;
    device const uint*  cg   = codes_g  + row * PC;
    device const uint*  cu   = codes_u  + row * PC;
    device const half*  sg   = scales_g + row * (C >> 7);
    device const half*  su   = scales_u + row * (C >> 7);
    device const half4* xr4  = (device const half4*)(x + (ulong)n * C);
    device const half4* cb4  = (device const half4*)cb;
    const uint nsub = C >> 3;
    float ag = 0.0f, au = 0.0f;
    for (uint k = lane; k < nsub; k += 32u) {
        uint ig, iu;
        if (nbits == 8u) {
            ig = (cg[k >> 2] >> ((k & 3u) * 8u)) & 255u;
            iu = (cu[k >> 2] >> ((k & 3u) * 8u)) & 255u;
        } else {
            uint pos = k * 12u; uint w = pos >> 5; uint off = pos & 31u;
            uint vg = cg[w] >> off; uint vu = cu[w] >> off;
            if (off > 20u) { vg |= cg[w + 1u] << (32u - off); vu |= cu[w + 1u] << (32u - off); }
            ig = vg & 4095u; iu = vu & 4095u;
        }
        half4 x0 = xr4[k * 2u];
        half4 x1 = xr4[k * 2u + 1u];
        half4 g0 = cb4[ig * 2u];
        half4 g1 = cb4[ig * 2u + 1u];
        half4 u0 = cb4[iu * 2u];
        half4 u1 = cb4[iu * 2u + 1u];
        float dg = (float)g0.x * (float)x0.x + (float)g0.y * (float)x0.y
                 + (float)g0.z * (float)x0.z + (float)g0.w * (float)x0.w
                 + (float)g1.x * (float)x1.x + (float)g1.y * (float)x1.y
                 + (float)g1.z * (float)x1.z + (float)g1.w * (float)x1.w;
        float du = (float)u0.x * (float)x0.x + (float)u0.y * (float)x0.y
                 + (float)u0.z * (float)x0.z + (float)u0.w * (float)x0.w
                 + (float)u1.x * (float)x1.x + (float)u1.y * (float)x1.y
                 + (float)u1.z * (float)x1.z + (float)u1.w * (float)x1.w;
        ag += (float)sg[k >> 4] * dg;
        au += (float)su[k >> 4] * du;
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0u) {
        float act = ag / (1.0f + metal::exp(-ag));    // silu(gate)
        y[(ulong)n * R + r] = (half)(act * au);
    }
"""

_KERNEL3_D8 = mx.fast.metal_kernel(
    name="vq_swiglu_d8",
    input_names=["x", "codes_g", "scales_g", "codes_u", "scales_u", "cb", "eidx", "params"],
    output_names=["y"],
    source=_SRC3_D8,
)


def vq_swiglu_d8(x_f16, cg_u32, sg_f16, cu_u32, su_f16, cb_f16, eidx_i32, nbits):
    """d=8 fused silu(gate(x)) * up(x) -> intermediate [N, R] f16."""
    N, C = x_f16.shape
    E, R, PC = cg_u32.shape
    params = mx.array([C, R, 8, nbits, PC, C // 128], dtype=mx.int32)
    (y,) = _KERNEL3_D8(
        inputs=[x_f16, cg_u32, sg_f16, cu_u32, su_f16, cb_f16, eidx_i32, params],
        output_shapes=[(N, R)],
        output_dtypes=[mx.float16],
        grid=(R * 32, N, 1),
        threadgroup=(256, 1, 1),
    )
    return y
