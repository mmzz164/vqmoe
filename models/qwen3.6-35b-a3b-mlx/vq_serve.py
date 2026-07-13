#!/usr/bin/env python
"""OpenAI-compatible server for the VQ (map12) model — mlx_lm.server with the VQ loader.

Usage:  python vq_serve.py --model ~/models/Qwen3.6-35B-A3B-MLX-VQ-map12 --port 8090
Then:   curl http://127.0.0.1:8090/v1/chat/completions -d '{"model":"vq","messages":[...]}'

Any model dir whose config.json has a "vq" section loads through vq_switch.load_vq_model
(VQSwitchLinear experts + Metal kernel); everything else falls back to stock mlx_lm.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VQ_KERNEL", "2")
os.environ.setdefault("VQ_FUSED", "1")

import mlx_lm.server as S
from mlx_lm.utils import load_tokenizer
from vq_switch import load_vq_model

_stock_load = S.load


def _vq_load(model_path, *args, **kw):
    p = str(model_path)
    cfg_path = os.path.join(p, "config.json")
    if os.path.isdir(p) and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            if "vq" in json.load(f):
                model, _config = load_vq_model(p)
                tok = load_tokenizer(Path(p), kw.get("tokenizer_config") or {})
                return model, tok
    return _stock_load(model_path, *args, **kw)


S.load = _vq_load

# A client that drops mid-stream (Claude Code Esc, curl timeout, proxy restart)
# makes the generation loop's socket writes raise BrokenPipe, which has been
# observed to take the whole server down. Contain it: stop that generation only.
_orig_handle_completion = S.APIHandler.handle_completion


def _safe_handle_completion(self, request, stop_words):
    try:
        _orig_handle_completion(self, request, stop_words)
    except (BrokenPipeError, ConnectionResetError) as e:
        import logging
        logging.warning(f"client disconnected mid-generation ({e!r}); request dropped")


S.APIHandler.handle_completion = _safe_handle_completion


# ---- persistent prompt cache: segment-boundary KV entries survive restarts ----
# mlx_lm.server stores prompt-cache entries at chat segment boundaries (system /
# user / assistant) in an LRUPromptCache. The system+tools segment is byte-stable
# across Claude Code sessions, so persisting those entries to disk turns the
# first ~35k tokens of every fresh session into a cache hit instead of a
# multi-minute prefill. VQ_CACHE=0 disables; VQ_CACHE_DIR / VQ_CACHE_DISK_GB tune.
if os.environ.get("VQ_CACHE", "1") != "0":
    import glob as _glob
    import hashlib as _hashlib
    import logging as _logging
    import queue as _queue
    import threading as _threading

    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache

    _CACHE_DIR = os.environ.get("VQ_CACHE_DIR", os.path.expanduser("~/.vq3/prompt_cache"))
    _CACHE_BYTES = float(os.environ.get("VQ_CACHE_DISK_GB", "8")) * 1e9

    class _PersistentLRUPromptCache(S.LRUPromptCache):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            os.makedirs(_CACHE_DIR, exist_ok=True)
            self._wq = _queue.Queue()
            _threading.Thread(target=self._writer, daemon=True).start()
            self._preload()

        @staticmethod
        def _fname(model, tokens):
            h = _hashlib.sha1()
            h.update(repr(model).encode())
            h.update(json.dumps(list(tokens)).encode())
            return h.hexdigest()[:20] + ".safetensors"

        def _preload(self):
            files = sorted(_glob.glob(os.path.join(_CACHE_DIR, "*.safetensors")),
                           key=os.path.getmtime, reverse=True)
            used, n = 0, 0
            for f in files:
                sz = os.path.getsize(f)
                if used + sz > _CACHE_BYTES:
                    try:
                        os.remove(f)                      # over budget: oldest go
                    except OSError:
                        pass
                    continue
                try:
                    cache, meta = load_prompt_cache(f, return_metadata=True)
                    model = tuple(json.loads(meta["model_key"]))
                    tokens = json.loads(meta["tokens"])
                    super().insert_cache(model, tokens, cache,
                                         cache_type=meta.get("cache_type", "system"))
                    used += sz
                    n += 1
                except Exception as e:
                    _logging.warning(f"[vq cache] dropping unreadable "
                                     f"{os.path.basename(f)}: {e!r}")
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            if n:
                _logging.info(f"[vq cache] preloaded {n} prompt-cache entries "
                              f"({used/1e9:.2f} GB) from {_CACHE_DIR}")

        def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
            super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
            if cache_type != "system":
                return    # user/assistant entries embed volatile turn text; churny
            path = os.path.join(_CACHE_DIR, self._fname(model, tokens))
            if not os.path.exists(path):
                # Materialize on THIS thread (which owns the generator's local
                # stream) — lazy arrays scheduled on a thread-local stream cannot
                # be evaluated from the writer thread. Stored cache objects are
                # never mutated after insertion (fetch deepcopies), so handing
                # references across threads is safe once evaluated.
                import mlx.core as mx
                mx.eval([c.state for c in prompt_cache])
                self._wq.put((path, model, list(tokens), prompt_cache, cache_type))

        def _writer(self):
            import mlx.core as mx
            mx.set_default_stream(mx.new_stream(mx.default_device()))
            while True:
                path, model, tokens, cache, ctype = self._wq.get()
                try:
                    meta = {"model_key": json.dumps([None if x is None else str(x)
                                                     for x in model]),
                            "tokens": json.dumps(tokens),
                            "cache_type": str(ctype)}
                    save_prompt_cache(path, cache, meta)
                    _logging.info(f"[vq cache] persisted {ctype} segment "
                                  f"({len(tokens)} tokens) -> {os.path.basename(path)}")
                except Exception as e:
                    _logging.warning(f"[vq cache] persist failed: {e!r}")
                    try:
                        os.remove(path)                    # no 0-byte tombstones
                    except OSError:
                        pass

    S.LRUPromptCache = _PersistentLRUPromptCache

if __name__ == "__main__":
    # Bigger prefill steps amortize the fast path's per-chunk expert dequant
    # (309 -> 438 tok/s at 8192 on the 2.4bpw build) BUT the per-chunk transient
    # scales with the step: 8192 spiked >21 GB and OOM-crashed a 48 GB Mac whose
    # Metal working-set limit is 40 GB once the prompt cache had grown to ~8 GB.
    # 2048 (stock mlx_lm default) keeps the transient ~5 GB; the headroom guard in
    # vq_switch is the backstop. Machines with more RAM can raise both VQ_PREFILL_STEP
    # and VQ_PREFILL_MIN_HEADROOM_GB together.
    if not any(a.startswith("--prefill-step-size") for a in sys.argv):
        sys.argv += ["--prefill-step-size", os.environ.get("VQ_PREFILL_STEP", "2048")]
    # Cap the KV prompt cache. On a 48 GB Mac it grew to 9.6 GB across cached
    # conversations and, stacked on the 10.5 GB model + a big-context prefill,
    # overflowed the Metal working set into the uncatchable OOM abort. Capping the
    # cache keeps the resident baseline low so big prompts have room. Persisted
    # system segments still preload (they fit under the cap). VQ_CACHE_RAM_GB tunes;
    # raise it on machines with more unified memory.
    if not any(a.startswith("--prompt-cache-bytes") for a in sys.argv):
        ram_gb = float(os.environ.get("VQ_CACHE_RAM_GB", "4"))
        sys.argv += ["--prompt-cache-bytes", str(int(ram_gb * 1024**3))]
    S.main()
