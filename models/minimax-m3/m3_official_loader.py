# SPDX-License-Identifier: Apache-2.0
"""Load OUR AutoRound 3.2bit (mixed_gptq) checkpoint into the OFFICIAL vLLM M3
text backbone (vLLM 0.23.1 nightly, native MiniMax-M3 + MSA indexer).

Our checkpoint was exported from the *transformers* MiniMax-M3-VL model, so its
keys are "reverse-normalized" relative to the original MiniMax HF naming that
the official vLLM model expects:

  ours (transformers)                      official vLLM CausalLM param
  ---------------------------------------  ----------------------------------
  model.language_model.layers.N.<X>        model.layers.N.<X'>
  ...mlp.experts.E.{gate,up,down}_proj     ...block_sparse_moe.experts.E.{w1,w3,w2}
  ...mlp.gate.weight                       ...block_sparse_moe.gate.weight
  ...mlp.gate.e_score_correction_bias      ...block_sparse_moe.e_score_correction_bias
  ...mlp.shared_experts.<X>                ...block_sparse_moe.shared_experts.<X>
  ...self_attn.indexer.<X>                 ...self_attn.index_<X>
  model.language_model.lm_head.weight      lm_head.weight

Two STRUCTURAL mismatches are resolved by selectively de-quantizing the
affected tensors to bf16 at load time (everything else stays quantized):

  1. Sparse layers fuse q/k/v/index_q/index_k into ONE quantized GEMM
     (MinimaxM3QKVParallelLinearWithIndexer). Our q/k/v are GPTQ-quantized but
     the indexer q/k are bf16 -> a single GPTQ linear can't mix precisions.
     Fix: de-quant q/k/v to bf16; feed all 5 shards bf16 (config marks
     self_attn.qkv_proj unquantized, see m3_quant.py).
  2. Dense MLP (layers 0-2) and the shared expert store gate_up *pre-fused*
     (one quantized tensor); the official MergedColumnParallelLinear wants
     SEPARATE gate_proj/up_proj shards. Fix: de-quant the fused tensor to bf16,
     split along the output dim, feed gate_proj/up_proj (config marks
     gate_up_proj unquantized).

The ~413B-param routed experts stay fully quantized and load through the stock
FusedMoE expert path: our MixedGPTQMoEMethod.create_weights registers the
standard placeholder params (experts.routed_experts.w13_*/w2_*) that
fused_moe_make_expert_params_mapping targets, and its mixed_weight_loader's
(param, loaded_weight, weight_name, shard_id, expert_id) signature matches how
the official MiniMaxM3Model.load_weights drives it. So routed experts need ONLY
key translation -- no de-quant, no numeric change.

De-quant uses gptq_v1_dequant (the PPL-validated reference), so the bf16 values
are bit-identical to the offline reference; this migration does not lose
accuracy beyond what the quantization already cost.

Vision is skipped: we force architectures=["MiniMaxM3SparseForCausalLM"] (the
text-only backbone) in serve_m3_official.py, so there is no vision tower to
load. The vision-capable path remains the out-of-tree M1 model.
"""
import os
import re
import sys

import torch

sys.path.insert(0, "/home/mizugaihiros01/work/onecomp")
sys.path.insert(0, "/var/hf/vllm_m3")

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.registry import ModelRegistry
from vllm_plugins.gptq.mixed_moe import gptq_v1_dequant

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# DIAGNOSTIC (M3_FULL_ATTN=1): replace the sparse layers' indexer + Triton
# block-sparse attend with a reference full causal GQA attention computed in
# pure PyTorch from the paged K/V cache. Decisive bisection:
#   * output -> "Paris" with LOW cos(full,orig) => the Triton sparse kernel
#     (minimax_m3_sparse_attn, selected on sm_120 since the MSA fmha_sm100 path
#     is SM100-only) is the broken kernel; full attention fixes it.
#   * output -> garbage with HIGH cos(full,orig) => attention is fine (and the
#     gather is validated by the agreement); the bug is upstream (the fused
#     qknorm-rope-kv-insert kernel, or fused_allreduce_gemma_rms_norm).
# At <=2048 ctx the indexer selects all blocks, so a correct sparse kernel MUST
# equal full attention -> any divergence is the kernel's numerics on Blackwell.
_FA_LOGGED: dict[str, bool] = {}


def _full_attention(self, query, output):
    """Reference full causal GQA attention over the paged K/V cache.

    query: [num_tokens, num_heads*head_dim] (already qk-normed + roped by the
    fused kernel). self.kv_cache: (num_blocks, 2, block_size, num_kv_heads,
    head_dim); k/v were just scatter-inserted by the fused kernel. Writes the
    attention result into `output` (same shape as query)."""
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):
        return output  # profiling run; caches unbound
    main_md = md[self.layer_name]
    nt = main_md.num_actual_tokens
    H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_dim
    scale = self.scaling
    rep = H // Hkv
    q = query[:nt].view(-1, H, D)
    out = output[:nt].view(-1, H, D)
    kc, vc = self.kv_cache.unbind(1)              # (nb, bs, Hkv, D) each
    bs = self.kv_cache.size(2)
    kflat = kc.reshape(-1, Hkv, D)                # (nb*bs, Hkv, D)
    vflat = vc.reshape(-1, Hkv, D)
    nd = main_md.num_decode_tokens
    dev = q.device
    ar = torch.arange(bs, device=dev)

    def attend(qj, slots, base):
        # qj: (Lq, H, D); slots: (Lk,) into kflat; base: key index of qj[0].
        k = kflat[slots].repeat_interleave(rep, dim=1).float()   # (Lk, H, D)
        v = vflat[slots].repeat_interleave(rep, dim=1).float()
        Lq, Lk = qj.shape[0], slots.shape[0]
        s = torch.einsum("qhd,khd->hqk", qj.float(), k) * scale   # (H, Lq, Lk)
        col = torch.arange(Lk, device=dev)[None, :]
        row = (base + torch.arange(Lq, device=dev))[:, None]
        s = s.masked_fill((col > row)[None], float("-inf"))       # causal
        a = torch.softmax(s, dim=-1)
        o = torch.einsum("hqk,khd->qhd", a, v)                    # (Lq, H, D)
        return o.to(out.dtype)

    if main_md.num_decodes > 0:
        d = main_md.decode
        for i in range(main_md.num_decodes):
            L = int(d.seq_lens[i].item())
            nblk = (L + bs - 1) // bs
            slots = (d.block_table[i, :nblk, None] * bs + ar[None, :]).reshape(-1)[:L]
            out[i : i + 1] = attend(q[i : i + 1], slots, L - 1)
    if main_md.num_prefills > 0:
        p = main_md.prefill
        for j in range(main_md.num_prefills):
            qs = int(p.cu_seqlens_q[j].item())
            qe = int(p.cu_seqlens_q[j + 1].item())
            qlen = qe - qs
            L = int(p.seq_lens[j].item())
            nblk = (L + bs - 1) // bs
            slots = (p.block_table[j, :nblk, None] * bs + ar[None, :]).reshape(-1)[:L]
            out[nd + qs : nd + qe] = attend(q[nd + qs : nd + qe], slots, L - qlen)
    return output


# ---------------------------------------------------------------------------
# DIAGNOSTIC (M3_PT_NORM=1): replace the per-layer norm + all-reduce subsystem
# with pure-PyTorch references. The official model uses two custom kernels at
# EVERY layer that M1 does NOT: (a) flashinfer Gemma RMSNorm (gemma_rmsnorm /
# gemma_fused_add_rmsnorm) and (b) fused_allreduce_gemma_rms_norm, whose fast
# path is a flashinfer TRT-LLM fused all-reduce+norm kernel. Either could be
# numerically wrong on Blackwell sm_120 and would corrupt the residual stream
# every layer -> wash out to noise over 60 layers. Replacing both with
# PyTorch (1+w) Gemma norm + explicit NCCL all-reduce (the M1-validated path)
# tests whether the bug lives in this subsystem.
_PTN_LOGGED = {"fi": False}


def _pt_gemma_norm(x, weight, eps):
    xf = x.float()
    xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xn * (1.0 + weight.float())).to(x.dtype)


def _gemma_forward(self, x, residual=None):
    """PyTorch replacement for MiniMAXGemmaRMSNorm.forward (flashinfer-free)."""
    if residual is None:
        return _pt_gemma_norm(x, self.weight, self.variance_epsilon)
    new_res = x + residual                       # gemma_fused_add: residual += x
    out = _pt_gemma_norm(new_res, self.weight, self.variance_epsilon)
    return out, new_res                          # (normed, summed) — matches flashinfer


def _pt_fused_allreduce_gemma_norm(hidden_states, residual, norm):
    """PyTorch replacement for fused_allreduce_gemma_rms_norm: explicit NCCL
    all-reduce of the partial + PyTorch Gemma add-norm. Logs once whether the
    flashinfer fused fast path WOULD have been taken (the prime suspect)."""
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
    if not _PTN_LOGGED["fi"]:
        _PTN_LOGGED["fi"] = True
        try:
            from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
                _can_use_flashinfer,
            )
            tp = get_tensor_model_parallel_world_size()
            ok, _ = _can_use_flashinfer(hidden_states, tp)
            logger.warning("M3_PTN: flashinfer fused-allreduce fast path would be "
                           "ACTIVE=%s (tp=%d) -> now forced to NCCL+pytorch", ok, tp)
        except Exception as e:
            logger.warning("M3_PTN flashinfer-probe failed: %r", e)
    if get_tensor_model_parallel_world_size() == 1:
        return _gemma_forward(norm, hidden_states, residual)
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return _gemma_forward(norm, reduced, residual)


# ---------------------------------------------------------------------------
# DIAGNOSTIC (M3_DENSE_FULL=1): replace the DENSE layers' (0-2) standard vLLM
# Attention (TRITON_ATTN backend on sm_120 -- the only place ② uses a stock
# attention backend that M1 does not) with a direct full causal GQA SDPA on the
# prefill q/k/v, and log cos vs the backend output. The dense layers are early,
# so if their attention is wrong on Blackwell the whole residual stream is
# corrupted from the start.
# ---------------------------------------------------------------------------
# DIAGNOSTIC (M3_MOE_DIAG=1): with attention/norm/activation all proven correct,
# the only remaining ②-specific suspects are MoE routing (wrong expert
# selection) or expert-weight loading. Log the router for the first few MoE
# layers: a degenerate router (uniform/near-zero logits) => gate weight / scoring
# bug; a healthy router (clear top experts) => suspicion shifts to expert compute.
_MOE_DIAG = {"n": 0}


def _moe_forward_diag(self, hidden_states):
    num_tokens, hidden_dim = hidden_states.shape
    hs = hidden_states.view(-1, hidden_dim)
    router_logits, _ = self.gate(hs)
    out = self.experts(hidden_states=hs, router_logits=router_logits)
    # On the FIRST MoE layer (3), first prefill, dump the internals so the
    # standalone HF reference (dbg_moe_ref.py) can localize router vs experts.
    if _MOE_DIAG["n"] == 0 and 2 <= num_tokens <= 64:
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank() == 0:
                bias = getattr(self, "e_score_correction_bias", None)
                torch.save({
                    "moe_in": hs.detach().float().cpu(),
                    "router_logits": router_logits.detach().float().cpu(),
                    "moe_out": out.detach().float().cpu(),
                    "ebias": (bias.detach().float().cpu() if bias is not None
                              else torch.zeros(router_logits.shape[-1])),
                }, "/tmp/m3dbg_OFFICIAL_moe3.pt")
                logger.warning("M3_MOE: dumped layer-3 internals to "
                               "/tmp/m3dbg_OFFICIAL_moe3.pt (T=%d H=%d)",
                               hs.shape[0], hs.shape[1])
        except Exception as e:
            logger.warning("M3_MOE dump failed: %r", e)
    if _MOE_DIAG["n"] < 3 and 2 <= num_tokens <= 64:
        _MOE_DIAG["n"] += 1
        rl = router_logits[-1].float()              # last prompt token
        top = torch.topk(rl, 8)
        sg = torch.sigmoid(rl)
        sgt = torch.topk(sg, 8)
        bias = getattr(self, "e_score_correction_bias", None)
        logger.warning(
            "M3_MOE#%d router_logits: norm=%.2f mean=%.3f std=%.3f min=%.2f max=%.2f | "
            "top8 ids=%s vals=%s | sigmoid top8=%s | bias=%s",
            _MOE_DIAG["n"], rl.norm().item(), rl.mean().item(), rl.std().item(),
            rl.min().item(), rl.max().item(), top.indices.tolist(),
            [round(v, 2) for v in top.values.tolist()],
            [round(v, 3) for v in sgt.values.tolist()],
            ("None" if bias is None else
             f"norm={bias.float().norm().item():.2f} absmax={bias.float().abs().max().item():.3f}"))
        logger.warning(
            "M3_MOE#%d in_norm=%.2f out_norm=%.2f out_absmax=%.2f scaling=%s",
            _MOE_DIAG["n"], hs[-1].float().norm().item(), out[-1].float().norm().item(),
            out[-1].float().abs().max().item(),
            getattr(self, "routed_scaling_factor", "?"))
        # Shared-expert presence/wiring check.
        se = getattr(self, "shared_experts", None)
        nse = getattr(self, "n_shared_experts", "<NA>")
        exp_se = getattr(self.experts, "shared_experts", "<noattr>")
        shared_norm = -1.0
        if se is not None:
            try:
                shared_norm = se(hs)[-1].float().norm().item()
            except Exception as e:
                logger.warning("M3_MOE#%d shared() call failed: %r", _MOE_DIAG["n"], e)
        logger.warning(
            "M3_MOE#%d n_shared_experts=%s self.shared_experts=%s experts.shared_experts=%s "
            "shared_call_norm=%.2f", _MOE_DIAG["n"], nse,
            type(se).__name__ if se is not None else None,
            type(exp_se).__name__ if exp_se not in (None, "<noattr>") else exp_se,
            shared_norm)
    return out.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# THE FIX (M3_FIX_SHARED=1): the official MiniMaxM3MoE passes shared_experts to
# FusedMoE expecting the quant method to fuse the shared-expert output, but our
# MixedGPTQMoEMethod only computes the routed experts -> the runner gets
# shared_output=None and DROPS the shared expert in EVERY MoE layer (3-59) ->
# the residual stream loses the shared contribution each layer -> washes out to
# garbage. Proven by dbg_moe_ref_official.py: official moe_out == routed*2 with
# the shared expert (norm 338 of 540) entirely missing. Fix = compute the shared
# expert explicitly and add it (M1 does exactly this). ②'s shared_experts has
# reduce_results=False (sharded partial) so it needs an all-reduce; the routed
# output from self.experts is already EP/TP-reduced.
def _moe_forward_fixed(self, hidden_states):
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
    num_tokens, hidden_dim = hidden_states.shape
    hs = hidden_states.view(-1, hidden_dim)
    router_logits, _ = self.gate(hs)
    out = self.experts(hidden_states=hs, router_logits=router_logits)  # routed*2, reduced
    if getattr(self, "shared_experts", None) is not None:
        shared = self.shared_experts(hs)                  # partial (reduce_results=False)
        if get_tensor_model_parallel_world_size() > 1:
            shared = tensor_model_parallel_all_reduce(shared)
        out = out + shared
    return out.view(num_tokens, hidden_dim)


def _dense_attn_forward(self, positions, hidden_states):
    from vllm import _custom_ops as ops
    qkv, _ = self.qkv_proj(hidden_states)
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, self.q_norm.weight, self.k_norm.weight,
        self.rotary_emb.cos_sin_cache, positions,
        self.num_heads, self.num_kv_heads, self.rotary_emb.rotary_dim,
        self.q_norm.variance_epsilon,
    )
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # Always run the backend (populates the KV cache for decode); get its output.
    orig = self.attn(q, k, v)
    attn_output = orig
    md = get_forward_context().attn_metadata
    H, Hkv, D, N = self.num_heads, self.num_kv_heads, self.head_dim, q.shape[0]
    # Single-seq prefill only (max_num_seqs=1): direct causal GQA over q/k/v.
    if isinstance(md, dict) and 2 <= N <= 4096:
        rep = H // Hkv
        qh = q.view(N, H, D).float()
        kh = k.view(N, Hkv, D).repeat_interleave(rep, dim=1).float()
        vh = v.view(N, Hkv, D).repeat_interleave(rep, dim=1).float()
        s = torch.einsum("qhd,khd->hqk", qh, kh) * self.scaling
        cmask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), 1)
        s = s.masked_fill(cmask[None], float("-inf"))
        full = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vh)
        full = full.reshape(N, H * D).to(q.dtype)
        ln = getattr(self.attn, "layer_name", str(id(self)))
        if ln not in _FA_LOGGED:
            _FA_LOGGED[ln] = True
            af, bf = full.float().flatten(), orig.float().flatten()
            cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
            logger.warning("M3_DENSE %s: cos(full,orig)=%.4f full_norm=%.1f "
                           "orig_norm=%.1f", ln, cos, af.norm().item(), bf.norm().item())
        attn_output = full
    output, _ = self.o_proj(attn_output)
    return output


def _run_attention_full(self, query, index_query, output):
    """Drop-in for MiniMaxM3SparseAttention._run_attention that uses the full
    reference attention. When M3_FA_COMPARE=1, also runs the original
    indexer+sparse-kernel path into a scratch tensor and logs the cosine
    similarity once per layer (on the first multi-token prefill)."""
    orig = None
    if os.environ.get("M3_FA_COMPARE") == "1":
        orig = torch.empty_like(output)
        try:
            topk = self.indexer(index_query)
            self.impl.forward(self, query, self.kv_cache, topk, orig)
        except Exception as e:  # never break the forward on the compare path
            logger.warning("M3_FA orig path failed %s: %r", self.layer_name, e)
            orig = None
    _full_attention(self, query, output)
    if orig is not None and self.layer_name not in _FA_LOGGED:
        md = get_forward_context().attn_metadata
        if isinstance(md, dict) and md[self.layer_name].num_prefill_tokens >= 2:
            _FA_LOGGED[self.layer_name] = True
            a, b = output.float().flatten(), orig.float().flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            logger.warning(
                "M3_FA %s: cos(full,orig)=%.4f full_norm=%.1f orig_norm=%.1f",
                self.layer_name, cos, a.norm().item(), b.norm().item())
    return output

# Quantized attn projections that must be de-quanted (fused with bf16 indexer).
_DEQUANT_ATTN = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj)$")


def _is_fused_gateup(modpath: str) -> bool:
    """Pre-fused gate_up (dense MLP or shared expert), NOT routed experts."""
    return modpath.endswith(".mlp.gate_up_proj") or modpath.endswith(
        ".shared_experts.gate_up_proj"
    )


def _is_shared_down(modpath: str) -> bool:
    """Shared-expert down_proj. The checkpoint stores it QUANTIZED, but the
    official model treats it as UNQUANTIZED: the per-module bits in
    quantization_config are keyed by the checkpoint suffix
    ``mlp.shared_experts.down_proj`` while get_quant_method is called with the
    official prefix ``block_sparse_moe.shared_experts.down_proj`` -> the bits
    lookup misses -> UnquantizedLinearMethod -> the param is ``.weight``. So we
    must de-quant it to bf16 (like the shared gate_up) or it never loads (stays
    zero) and the whole shared expert outputs 0 in every MoE layer -> garbage."""
    return modpath.endswith(".shared_experts.down_proj")


def _translate_layer(rest: str) -> str:
    """Translate the part after `layers.N.` (ours -> official CausalLM)."""
    # MSA lightning indexer:  self_attn.indexer.<X> -> self_attn.index_<X>
    rest = re.sub(r"^self_attn\.indexer\.", "self_attn.index_", rest)

    # routed experts: mlp.experts.E.{gate,up,down}_proj.<suf> ->
    #                 block_sparse_moe.experts.E.{w1,w3,w2}.<suf>
    m = re.match(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(.+)$", rest)
    if m:
        role = {"gate": "w1", "up": "w3", "down": "w2"}[m.group(2)]
        return f"block_sparse_moe.experts.{m.group(1)}.{role}.{m.group(3)}"

    # router bias / weight
    if rest == "mlp.gate.e_score_correction_bias":
        return "block_sparse_moe.e_score_correction_bias"
    if rest == "mlp.gate.weight":
        return "block_sparse_moe.gate.weight"

    # shared expert:  mlp.shared_experts.<X> -> block_sparse_moe.shared_experts.<X>
    if rest.startswith("mlp.shared_experts."):
        return "block_sparse_moe." + rest[len("mlp.") :]

    # dense MLP (layers 0-2) mlp.gate_up_proj / mlp.down_proj: keep mlp.*
    # self_attn.{q,k,v,o}_proj / q_norm / k_norm / index_*_norm,
    # input_layernorm, post_attention_layernorm: unchanged.
    return rest


def _translate_key(key: str):
    """Ours -> official CausalLM param name. None => skip (vision)."""
    if "vision_tower" in key or "multi_modal_projector" in key:
        return None
    # lm_head: the transformers VL checkpoint stores it as
    # `language_model.lm_head.weight` (NOT under model.language_model); the
    # text-only CausalLM expects top-level `lm_head.weight`. (tie_word_embeddings
    # is False, so this is a required separate weight -- missing it = random
    # logits = garbage output.)
    if key.startswith("model.language_model.lm_head."):
        return "lm_head." + key[len("model.language_model.lm_head.") :]
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key[len("language_model.lm_head.") :]
    if key.startswith("lm_head."):
        return key
    if key.startswith("model.language_model.embed_tokens."):
        return "model.embed_tokens." + key[len("model.language_model.embed_tokens.") :]
    if key.startswith("model.language_model.norm."):
        return "model.norm." + key[len("model.language_model.norm.") :]
    m = re.match(r"^model\.language_model\.layers\.(\d+)\.(.+)$", key)
    if m:
        return f"model.layers.{m.group(1)}." + _translate_layer(m.group(2))
    # Unknown top-level key (e.g. mtp.*) -> skip; the model's own loader also
    # ignores mtp. Anything genuinely required will surface as a missing param.
    logger.warning("m3_official_loader: skipping unrecognized key %s", key)
    return None


def _emit_dequant(modpath: str, parts: dict):
    """De-quant one buffered GPTQ module -> bf16; yield official (name, tensor).

    gptq_v1_dequant is the PPL-validated v1 decoder; it returns fp32 [out, in].
    bits/in/out are inferred from tensor shapes (AutoGPTQ v1 layout:
    qweight [in*bits/32, out], scales [G, out], qzeros [G, out*bits/32]).
    """
    qw, qz, sc = parts["qweight"], parts["qzeros"], parts["scales"]

    if modpath.endswith((".q_proj", ".k_proj", ".v_proj")):
        # q/k/v_proj fuse with the bf16 indexer -> must be bf16: de-quant the
        # single GPTQ tensor (gptq_v1_dequant = PPL-validated v1 decoder, fp32
        # [out, in]). v1 layout: qweight [in*bits/32, out], scales [G, out],
        # qzeros [G, out*bits/32].
        out_f = sc.shape[1]
        bits = (qz.shape[1] * 32) // out_f
        in_f = (qw.shape[0] * 32) // bits
        w = gptq_v1_dequant(qw, sc, qz, in_f, out_f, bits).to(torch.bfloat16)
        yield _translate_key(modpath + ".weight"), w
        return

    # fused gate_up -> split the QUANTIZED tensors along the OUTPUT dim into
    # separate gate_proj/up_proj GPTQ shards (NO de-quant -> stays quantized,
    # reclaims ~2 GiB/rank vs bf16). out = 2*I. qweight [in*bits/32, 2I] and
    # scales [G, 2I] split at I; qzeros [G, 2I*bits/32] split at qz.shape[1]//2
    # (= I*bits/32, a clean 32-bit word boundary for the 4/8-bit gate_up).
    I = sc.shape[1] // 2
    zh = qz.shape[1] // 2
    base = _translate_key(modpath + ".weight").replace("gate_up_proj.weight", "")
    for role, ws, zs in (
        ("gate_proj", slice(0, I), slice(0, zh)),
        ("up_proj", slice(I, None), slice(zh, None)),
    ):
        yield base + role + ".qweight", qw[:, ws].contiguous()
        yield base + role + ".scales", sc[:, ws].contiguous()
        yield base + role + ".qzeros", qz[:, zs].contiguous()


def translate_and_dequant(weights):
    """Stream our checkpoint weights -> official-named weights.

    De-quant targets (q/k/v_proj, fused gate_up) are buffered until their three
    GPTQ sub-tensors (qweight/qzeros/scales) are all present, then de-quanted and
    emitted. Within a safetensors shard these keys are adjacent, so the buffer
    holds ~one module at a time. Everything else is translated and passed through
    unchanged (routed experts and o_proj/down_proj stay quantized).
    """
    buf: dict[str, dict] = {}
    for name, w in weights:
        if "vision_tower" in name or "multi_modal_projector" in name:
            continue
        modpath, _, suffix = name.rpartition(".")
        if suffix in ("qweight", "qzeros", "scales") and (
            _DEQUANT_ATTN.search(modpath) or _is_fused_gateup(modpath)
        ):
            d = buf.setdefault(modpath, {})
            d[suffix] = w
            if len(d) == 3:
                yield from _emit_dequant(modpath, d)
                del buf[modpath]
            continue
        tk = _translate_key(name)
        if tk is not None:
            yield tk, w
    if buf:
        raise RuntimeError(
            f"m3_official_loader: incomplete GPTQ tensors for {list(buf)}"
        )


def register():
    """Subclass the text-only CausalLM, wrap load_weights with translation, and
    register it for the MiniMaxM3SparseForCausalLM arch (overrides the stock
    entry). Re-runs on every import incl. spawn workers -> propagates."""
    from vllm.models.minimax_m3 import MiniMaxM3SparseForCausalLM

    _diag = {"layers": 0}

    class MiniMaxM3SparseForCausalLM_OurQuant(MiniMaxM3SparseForCausalLM):
        def load_weights(self, weights):
            n = super().load_weights(translate_and_dequant(weights))
            if os.environ.get("M3_DIAG_HOOKS") == "1":
                self._install_diag_hooks()
            if os.environ.get("M3_MOE_DIAG") == "1":
                # Localize the missing shared expert: list the model's actual
                # shared-expert param names + post-load norms (0 => never loaded).
                import torch as _t
                shown = 0
                for pn, p in self.named_parameters():
                    if "shared_expert" in pn and (".3." in pn or ".layers.3." in pn):
                        nrm = p.float().norm().item() if p.numel() else -1
                        logger.warning("M3_SHARED_PARAM %s shape=%s norm=%.3f loaded=%s",
                                       pn, tuple(p.shape), nrm, pn in n)
                        shown += 1
                        if shown >= 12:
                            break
                logger.warning("M3_SHARED_PARAM total loaded-set size=%d", len(n))
            return n

        def compute_logits(self, hidden_states):
            logits = super().compute_logits(hidden_states)
            if (os.environ.get("M3_DIAG_HOOKS") == "1"
                    and logits is not None and not _diag.get("logits_done")):
                _diag["logits_done"] = True
                try:
                    import torch as _t
                    last = logits[-1].float()
                    top = _t.topk(last, 8)
                    logger.warning(
                        "M3_LOGITS last-position top8: ids=%s vals=%s",
                        top.indices.tolist(),
                        [round(v, 2) for v in top.values.tolist()])
                except Exception as e:
                    logger.warning("M3_LOGITS failed: %r", e)
            return logits

        def _install_diag_hooks(self):
            import torch as _t

            def mk(idx):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # ONLY the real prefill (seqlen 2..64): skip the 2048-token
                    # profiling AND the seqlen-1 warmup/decode forwards (the
                    # warmup has an unbound cache -> attention 0 -> misleading).
                    if not (2 <= h.shape[0] <= 64):
                        return
                    if _diag["layers"] >= 60:
                        return
                    _diag["layers"] += 1
                    r = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                    def st(t):
                        if t is None:
                            return "None"
                        tf = t.float()
                        return (f"norm={tf.norm():.1f} absmax={tf.abs().max():.2f} "
                                f"nan={int(_t.isnan(tf).sum())} inf={int(_t.isinf(tf).sum())}")
                    comb = -1.0
                    if isinstance(h, _t.Tensor) and isinstance(r, _t.Tensor):
                        comb = (h.float() + r.float()).norm().item()
                    logger.warning(
                        "M3_LAYER_DIAG L%02d seqlen=%d: hid[%s] resid[%s] COMB=%.1f",
                        idx, (h.shape[0] if isinstance(h, _t.Tensor) else -1),
                        st(h), st(r), comb)
                return hook

            # Submodule input hooks for the first 8 layers: the value going INTO
            # attention / FFN is the gemma-normed hidden state. If its RMS is ~1
            # the norm works; if it grows with depth the (fused) norm is broken.
            def mk_sub(idx, kind):
                def hook(mod, args, kwargs, out):
                    try:
                        x = kwargs.get("hidden_states")
                        if x is None and args:
                            x = args[1] if (kind == "attn" and len(args) > 1) else args[0]
                        if (not isinstance(x, _t.Tensor) or x.dim() < 2
                                or not (2 <= x.shape[0] <= 64)):
                            return
                        xf = x.float()
                        rms = xf.pow(2).mean().sqrt().item()
                        o = out[0] if isinstance(out, tuple) else out
                        of = o.float() if isinstance(o, _t.Tensor) else None
                        logger.warning(
                            "M3_SUB_DIAG L%02d %s: IN rms=%.3f absmax=%.2f | "
                            "OUT norm=%.1f absmax=%.2f", idx, kind, rms,
                            xf.abs().max().item(),
                            (of.norm().item() if of is not None else -1),
                            (of.abs().max().item() if of is not None else -1))
                    except Exception as e:  # never crash the engine
                        logger.warning("M3_SUB_DIAG L%02d %s failed: %r", idx, kind, e)
                return hook

            # Inner standard-Attention hook (dense layers only): does FLASH_ATTN
            # get valid q/k/v but return ~0 (-> backend/metadata bug) or is q
            # already zero (-> fused qknorm-rope kernel)?
            def mk_attn(idx):
                def hook(mod, args, kwargs, out):
                    try:
                        q = kwargs.get("query", args[0] if args else None)
                        k = kwargs.get("key", args[1] if len(args) > 1 else None)
                        v = kwargs.get("value", args[2] if len(args) > 2 else None)
                        if not isinstance(q, _t.Tensor) or q.shape[0] > 64:
                            return
                        o = out[0] if isinstance(out, tuple) else out
                        g = lambda t: (t.float().norm().item()
                                       if isinstance(t, _t.Tensor) else -1)
                        logger.warning(
                            "M3_INNER_ATTN L%02d seqlen=%d: q=%.2f k=%.2f v=%.2f"
                            " -> out norm=%.3f absmax=%.3f", idx, q.shape[0],
                            g(q), g(k), g(v), g(o),
                            (o.float().abs().max().item()
                             if isinstance(o, _t.Tensor) else -1))
                    except Exception as e:
                        logger.warning("M3_INNER_ATTN L%02d failed: %r", idx, e)
                return hook

            layers = self.model.layers
            for i, lyr in enumerate(layers):
                lyr.register_forward_hook(mk(i))
                if i < 3 and hasattr(lyr.self_attn, "attn"):
                    lyr.self_attn.attn.register_forward_hook(mk_attn(i),
                                                             with_kwargs=True)
                if i < 3 and hasattr(lyr.self_attn, "o_proj"):
                    def mk_op(idx):
                        def hook(mod, args, kwargs, out):
                            try:
                                x = kwargs.get("input_", args[0] if args else None)
                                if not isinstance(x, _t.Tensor) or x.shape[0] > 64:
                                    return
                                o = out[0] if isinstance(out, tuple) else out
                                logger.warning(
                                    "M3_OPROJ L%02d seqlen=%d: IN norm=%.2f -> "
                                    "OUT norm=%.3f", idx, x.shape[0],
                                    x.float().norm().item(),
                                    (o.float().norm().item()
                                     if isinstance(o, _t.Tensor) else -1))
                            except Exception as e:
                                logger.warning("M3_OPROJ L%02d failed: %r", idx, e)
                        return hook
                    lyr.self_attn.o_proj.register_forward_hook(mk_op(i),
                                                               with_kwargs=True)
                if i < 8:
                    lyr.self_attn.register_forward_hook(mk_sub(i, "attn"),
                                                        with_kwargs=True)
                    ffn = getattr(lyr, "block_sparse_moe", None) or getattr(lyr, "mlp", None)
                    if ffn is not None:
                        ffn.register_forward_hook(mk_sub(i, "ffn"), with_kwargs=True)
            logger.warning("M3_LAYER_DIAG: installed hooks on %d layers", len(layers))

    ModelRegistry.register_model(
        "MiniMaxM3SparseForCausalLM", MiniMaxM3SparseForCausalLM_OurQuant
    )
    logger.info("m3_official_loader: registered key-translating CausalLM loader")

    # DIAGNOSTIC bisection: swap the sparse attention for the full-attention
    # reference (gated; off for normal serving). Runs in every spawn worker.
    if os.environ.get("M3_FULL_ATTN") == "1":
        from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
        MiniMaxM3SparseAttention._run_attention = _run_attention_full
        logger.warning("m3_official_loader: M3_FULL_ATTN=1 -> sparse attention "
                       "replaced with full-attention reference")

    # NOTE: the shared expert is now fixed at the SOURCE (config n_shared_experts
    # via serve_m3_official's FORCE dict so the module is built + its weights
    # load -> the FusedMoE runner adds it). The old M3_FIX_SHARED monkeypatch
    # (_moe_forward_fixed) is intentionally NOT wired up: with the module present
    # the runner already adds shared, so an explicit add would DOUBLE-count it.

    # DIAGNOSTIC: log MoE router for the first few MoE layers.
    if os.environ.get("M3_MOE_DIAG") == "1":
        from vllm.models.minimax_m3.nvidia.model import MiniMaxM3MoE
        MiniMaxM3MoE.forward = _moe_forward_diag
        logger.warning("m3_official_loader: M3_MOE_DIAG=1 -> MoE router logging on")

    # DIAGNOSTIC: replace dense-layer attention with full-attention reference.
    if os.environ.get("M3_DENSE_FULL") == "1":
        from vllm.models.minimax_m3.nvidia.model import MiniMaxM3Attention
        MiniMaxM3Attention.forward = _dense_attn_forward
        logger.warning("m3_official_loader: M3_DENSE_FULL=1 -> dense attention "
                       "replaced with full-attention reference")

    # DIAGNOSTIC: replace the Gemma norm + all-reduce subsystem with PyTorch.
    if os.environ.get("M3_PT_NORM") == "1":
        import vllm.models.minimax_m3.nvidia.model as _m
        _m.MiniMAXGemmaRMSNorm.forward = _gemma_forward
        _m.fused_allreduce_gemma_rms_norm = _pt_fused_allreduce_gemma_norm
        logger.warning("m3_official_loader: M3_PT_NORM=1 -> Gemma RMSNorm + "
                       "fused all-reduce replaced with PyTorch/NCCL references")


register()
