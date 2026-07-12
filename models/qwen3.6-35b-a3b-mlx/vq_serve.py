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

if __name__ == "__main__":
    S.main()
