"""LZ-style (compression-inspired) anti-repetition logits processor for vLLM v1.

WHY (vs repetition_penalty): a token-level repetition/frequency penalty penalizes
*every* recurring token, including the digits that legitimately recur in arithmetic
("800 + 160 + 56"), which empirically corrupts the math (17x24 -> 348). This processor
instead penalizes a candidate token only in proportion to how much choosing it would
EXTEND a repeated subsequence already present in the recent window -- i.e. how
"compressible" (LZ77-like) the stream would become. A token that is novel in context
(the digits of a *first* derivation, a fresh "</think>") is left untouched; only the
tokens that continue a repeating phrase (the degenerate self-doubt loop) are suppressed.

This is an LZ-*inspired* approximation of arXiv:2504.20131 (not the exact LZ77 codelength):
for each n-gram scale, find earlier occurrences of the current (n-1)-token suffix in the
window; the tokens that followed those occurrences are "loop-continuation" tokens and get
penalty += n per occurrence (longer match & more repeats => larger penalty). Finally
logit[t] -= alpha * min(pen[t], cap).

Registered additively via LLM(logits_processors=[LZPenaltyLogitsProcessor]); the vLLM
package and the model weights are untouched. Knobs live in the module-level _CFG dict so a
sweep can change alpha between llm.generate() calls WITHOUT rebuilding the engine.

Env (read once at import; _CFG is mutable at runtime):
  GLM_LZ_PENALTY=1        enable
  GLM_LZ_ALPHA=0.6        strength (logit units per unit penalty)
  GLM_LZ_NGRAMS=4,8,16,32 match scales n (penalize continuation of the (n-1)-suffix)
  GLM_LZ_WINDOW=512       lookback window over OUTPUT tokens
  GLM_LZ_CAP=60           cap on raw per-token penalty before alpha
"""
import os
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


def _parse_ngrams(s):
    return sorted({int(x) for x in s.split(",") if x.strip()})


# Mutable so a sweep can retune between generate() calls without rebuilding the LLM.
_CFG = {
    "enabled": os.environ.get("GLM_LZ_PENALTY", "") == "1",
    "alpha": float(os.environ.get("GLM_LZ_ALPHA", "0.6")),
    "ngrams": _parse_ngrams(os.environ.get("GLM_LZ_NGRAMS", "4,8,16,32")),
    "window": int(os.environ.get("GLM_LZ_WINDOW", "512")),
    "cap": float(os.environ.get("GLM_LZ_CAP", "60")),
}


def _penalty(output_ids):
    """Return {token_id: raw_penalty} for continuation-of-repeat tokens, or {}."""
    ngrams = _CFG["ngrams"]
    if not ngrams:
        return {}
    window = _CFG["window"]
    w = output_ids[-window:] if len(output_ids) > window else output_ids
    Lw = len(w)
    pen = {}
    for n in ngrams:
        k = n - 1                      # suffix length to match
        if Lw <= k:                    # need at least k history + 1 continuation
            continue
        suffix = w[Lw - k:]            # current last-k tokens
        limit = Lw - k                 # p in [0, limit) so p+k <= Lw-1
        for p in range(limit):
            if w[p:p + k] == suffix:
                cont = w[p + k]
                pen[cont] = pen.get(cont, 0.0) + float(n)
    return pen


# --- Budget forcing: hard backstop for loops LZ can't catch (varied-phrasing doubt) ---
# Our think prompts end with an OPEN <think>, so output_ids == thinking tokens; once
# len(output_ids) >= budget and the end token has not been emitted, force </think>
# (logit -> +inf). Mutable so a sweep can change the budget without rebuilding the LLM.
_BCFG = {
    "budget": int(os.environ.get("GLM_BUDGET", "0")),     # <=0 disables
    "end_id": int(os.environ.get("GLM_THINK_END_ID", "154842")),  # GLM </think>
}


class BudgetForceLogitsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False  # forces a specific token -> changes argmax

    def new_req_logits_processor(self, params):
        def _lp(output_ids, logits):
            b = _BCFG["budget"]
            if b > 0 and len(output_ids) >= b and _BCFG["end_id"] not in output_ids:
                logits[_BCFG["end_id"]] = 1e9   # force </think> as the next token
            return logits

        return _lp


class LZPenaltyLogitsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        # It changes which token greedy picks -> must NOT be argmax-invariant,
        # else vLLM may skip it for temperature=0 requests.
        return False

    def new_req_logits_processor(self, params):
        # Always attach; the per-request closure no-ops live when disabled/alpha<=0.
        def _lp(output_ids, logits):
            if not _CFG["enabled"] or _CFG["alpha"] <= 0.0:
                return logits
            pen = _penalty(output_ids)
            if not pen:
                return logits
            cap = _CFG["cap"]
            idx = torch.tensor(list(pen.keys()), device=logits.device, dtype=torch.long)
            vals = torch.tensor([(-_CFG["alpha"]) * min(pen[t], cap) for t in pen],
                                device=logits.device, dtype=logits.dtype)
            logits.index_add_(0, idx, vals)
            return logits

        return _lp
