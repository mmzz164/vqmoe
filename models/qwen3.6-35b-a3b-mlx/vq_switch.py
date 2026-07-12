"""VQSwitchLinear — pure-MLX VQ (codebook) switch linear for Qwen3.6 map12 (v3, level 0).

Drop-in replacement for mlx_lm's QuantizedSwitchLinear on the switch_mlp projections.
Weights per module (loaded via model.load_weights):
  vq_codes  int32  [E, R, PC]   LSB-first packed nbits-bit codebook indices
  vq_scales float16[E, R, C/128] per-128-group std scales
Shared codebook cb [K, d] float16 is set post-load via set_codebook().

Call contract mirrors QuantizedSwitchLinear.__call__(x, indices, sorted_indices):
  x [..., 1, C] broadcast against indices [..., k] -> y [..., k, R] (matching gather_qmm).
Level 0 = decode active experts on the fly (correctness first; Metal kernel later).

Also provides load_vq_model(path): builds the mlx_lm qwen3_5_moe model, quantizes the
spine per config, swaps switch_mlp projections for VQSwitchLinear, strict-loads the
artifact, installs codebooks. Needs mlx_lm >= 0.31 (qwen3_5_moe).
"""
import glob
import json
import math
import os

import mlx.core as mx
import mlx.nn as nn


def _unpack_plan(nbits, nsub):
    """Per-subvector (word, off, need_hi) for LSB-first nbits packing into 32-bit words."""
    words, offs, hi = [], [], []
    for k in range(nsub):
        pos = k * nbits
        words.append(pos // 32)
        offs.append(pos % 32)
        hi.append(1 if (pos % 32) + nbits > 32 else 0)
    return words, offs, hi


class VQSwitchLinear(nn.Module):
    def __init__(self, num_experts, output_dims, input_dims, nbits, d, norm_group=128):
        super().__init__()
        nsub = input_dims // d
        pc = (nsub * nbits + 31) // 32
        self.vq_codes = mx.zeros((num_experts, output_dims, pc), dtype=mx.int32)
        self.vq_scales = mx.zeros((num_experts, output_dims, input_dims // norm_group),
                                  dtype=mx.float16)
        self._nbits, self._d, self._C, self._ng = nbits, d, input_dims, norm_group
        w, o, h = _unpack_plan(nbits, nsub)
        self._uw = mx.array(w, dtype=mx.int32)          # [nsub] word index
        self._uo = mx.array(o, dtype=mx.uint32)         # [nsub] bit offset
        self._uh = mx.array(h, dtype=mx.uint32)         # [nsub] straddles word boundary
        self._cb = None
        self.freeze()

    def set_codebook(self, cb):
        self._cb = cb.astype(mx.float16)                # [K, d]
        self._codes_u32 = mx.view(self.vq_codes, mx.uint32)
        self._sc16 = self.vq_scales.astype(mx.float16)
        kv = os.environ.get("VQ_KERNEL", "2")
        if kv in ("1", "2") and self._d == 4:
            import vq_kernel
            self._gemv = vq_kernel.vq_gemv2 if kv == "2" else vq_kernel.vq_gemv
        else:
            self._gemv = None

    @property
    def input_dims(self):
        return self._C

    @property
    def output_dims(self):
        return self.vq_codes.shape[1]

    @property
    def num_experts(self):
        return self.vq_codes.shape[0]

    def _unpack(self, packed):
        """packed [..., PC] int32 -> codes [..., nsub] int32 (logical shifts via uint32 view)."""
        u = mx.view(packed, mx.uint32)
        lo = mx.take(u, self._uw, axis=-1)              # [..., nsub]
        lo = mx.right_shift(lo, self._uo)
        pc = u.shape[-1]
        wnext = mx.minimum(self._uw + 1, pc - 1)
        hi = mx.take(u, wnext, axis=-1)
        hi = mx.left_shift(hi, 32 - self._uo) * self._uh
        codes = mx.bitwise_or(lo, hi) & ((1 << self._nbits) - 1)
        return codes.astype(mx.int32)

    def _decode(self, eidx):
        """eidx [N] int -> dequantized weights [N, R, C] float16."""
        codes = self._unpack(mx.take(self.vq_codes, eidx, axis=0))      # [N, R, nsub]
        sub = mx.take(self._cb, codes.reshape(-1), axis=0)              # [N*R*nsub, d]
        N = eidx.shape[0]
        R, nsub = codes.shape[1], codes.shape[2]
        sub = sub.reshape(N, R, nsub, self._d)
        sc = mx.take(self.vq_scales, eidx, axis=0)                      # [N, R, C/ng]
        sc = mx.repeat(sc, self._ng // self._d, axis=-1)                # [N, R, nsub]
        return (sub * sc[..., None]).reshape(N, R, self._C)

    def __call__(self, x, indices, sorted_indices=False):
        # x [..., 1, C]; indices [...]; returns [..., 1, R] per index position
        if getattr(self, "_gemv", None) is not None:
            flat = indices.reshape(-1).astype(mx.int32)
            xk = mx.broadcast_to(x, (*indices.shape, 1, self._C)).reshape(-1, self._C)
            y = self._gemv(xk.astype(mx.float16), self._codes_u32, self._sc16,
                           self._cb, flat, self._nbits)
            return y.reshape(*indices.shape, 1, -1).astype(x.dtype)
        flat = indices.reshape(-1)
        W = self._decode(flat)                                          # [N, R, C]
        W = W.reshape(*indices.shape, *W.shape[1:])                     # [..., R, C]
        y = mx.matmul(x.astype(W.dtype), mx.swapaxes(W, -1, -2))        # [..., 1|k, R]-broadcast
        return y


class VQSwitchGLU(nn.Module):
    """Fused replacement for mlx_lm SwitchGLU on VQ layers: one Metal dispatch for
    silu(gate)*up + one for down. Children keep SwitchGLU's names so load_weights
    resolves ...switch_mlp.{gate,up,down}_proj.vq_codes unchanged."""

    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def __call__(self, x, indices):
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
        from vq_kernel import vq_swiglu
        g, u, dn = self.gate_proj, self.up_proj, self.down_proj
        if getattr(dn, "_gemv", None) is None or getattr(g, "_cb", None) is None:
            # fallback: reference SwitchGLU flow through the modules (level-0)
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            idx, inv_order = indices, None
            if do_sort:
                x, idx, inv_order = _gather_sort(x, indices)
            xu = u(x, idx, sorted_indices=do_sort)
            xg = g(x, idx, sorted_indices=do_sort)
            y = dn(mx.sigmoid(xg) * xg * xu, idx, sorted_indices=do_sort)
            if do_sort:
                y = _scatter_unsort(y, inv_order, indices.shape)
            return y.squeeze(-2)
        dtype = x.dtype
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        flat = idx.reshape(-1).astype(mx.int32)
        xk = mx.broadcast_to(x, (*idx.shape, 1, g._C)).reshape(-1, g._C).astype(mx.float16)
        inter = vq_swiglu(xk, g._codes_u32, g._sc16, u._codes_u32, u._sc16,
                          g._cb, flat, g._nbits)                      # [N, R]
        y = dn._gemv(inter, dn._codes_u32, dn._sc16, dn._cb, flat, dn._nbits)  # [N, C]
        y = y.reshape(*idx.shape, 1, -1).astype(dtype)
        if do_sort:
            y = _scatter_unsort(y, inv_order, indices.shape)
        return y.squeeze(-2)


def load_vq_model(path):
    """Build qwen3_5_moe, quantize spine per config, swap switch_mlp -> VQ, load weights."""
    import importlib

    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    arch = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    model = arch.Model(arch.ModelArgs.from_dict(config))

    vq_meta = config["vq"]["modules"]
    quant = config.get("quantization", {})

    # 1) spine quantization exactly like mlx_lm.load_model
    def class_predicate(p, m):
        if p in vq_meta:
            return False                                  # VQ modules: skip nn.quantize
        if p in quant:
            return quant[p]
        if not hasattr(m, "to_quantized"):
            return False
        return True

    nn.quantize(model, group_size=quant.get("group_size", 64), bits=quant.get("bits", 4),
                mode=quant.get("mode", "affine"), class_predicate=class_predicate)

    # 2) swap switch_mlp projections, then the whole switch_mlp for the fused GLU
    glu_parents = {}
    for mpath, meta in vq_meta.items():
        parts = mpath.split(".")
        parent = model
        for q in parts[:-1]:
            parent = parent[int(q)] if q.isdigit() else getattr(parent, q)
        old = getattr(parent, parts[-1])
        vql = VQSwitchLinear(num_experts=256, output_dims=old.output_dims if hasattr(old, "output_dims") else old.weight.shape[1],
                             input_dims=meta["in_dims"], nbits=meta["nbits"], d=meta["d"],
                             norm_group=meta["norm_group"])
        setattr(parent, parts[-1], vql)
        glu_parents[".".join(parts[:-1])] = parent
    if os.environ.get("VQ_FUSED", "1") == "1":
        for gpath, sw in glu_parents.items():
            parts = gpath.split(".")
            gp = model
            for q in parts[:-1]:
                gp = gp[int(q)] if q.isdigit() else getattr(gp, q)
            setattr(gp, parts[-1], VQSwitchGLU(sw.gate_proj, sw.up_proj, sw.down_proj))

    # 3) strict load
    shards = sorted(glob.glob(os.path.join(path, "model-*.safetensors")))
    weights = {}
    for s in shards:
        weights.update(mx.load(s))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)

    # 4) codebooks
    cbs = mx.load(os.path.join(path, config["vq"]["codebooks_file"]))
    for mpath, meta in vq_meta.items():
        parts = mpath.split(".")
        parent = model
        for q in parts[:-1]:
            parent = parent[int(q)] if q.isdigit() else getattr(parent, q)
        getattr(parent, parts[-1]).set_codebook(cbs[f"cb{meta['vq_bits']}"])

    mx.eval(model.parameters())
    model.eval()
    return model, config
