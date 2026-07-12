"""End-to-end VQ integration test on the Mac: load_vq_model + real generation.

Run:  ~/work/llm/omlx/.venv/bin/python vq_generate.py <artifact_dir> [max_tokens]
Loads the full qwen3_5_moe model with VQSwitchLinear experts (Metal kernels +
fused swiglu by default; VQ_KERNEL=0 for the pure-MLX reference path), runs a
short Japanese generation, reports text + tok/s.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vq_switch import load_vq_model

ART = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/models/Qwen3.6-35B-A3B-MLX-VQ-map12")
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 40

t0 = time.time()
def log(m): print(f"[vqgen +{time.time()-t0:.0f}s] {m}", flush=True)

log(f"loading VQ model from {ART} ...")
model, config = load_vq_model(ART)
log("model loaded (strict) — VQ modules live")

from mlx_lm.utils import load_tokenizer
from pathlib import Path
tokenizer = load_tokenizer(Path(ART))
msgs = [{"role": "user", "content": "東京の観光名所を3つ、名前だけ簡潔に教えてください。"}]
prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
log(f"prompt tokens: {len(prompt)}")

from mlx_lm import generate
t1 = time.time()
out = generate(model, tokenizer, prompt=prompt, max_tokens=MAXTOK, verbose=False)
dt = time.time() - t1
mode = f"VQ_KERNEL={os.environ.get('VQ_KERNEL', '2')} VQ_FUSED={os.environ.get('VQ_FUSED', '1')}"
log(f"generated {MAXTOK} tokens in {dt:.1f}s ({MAXTOK/dt:.2f} tok/s, {mode})")
print("---- OUTPUT ----")
print(out)
print("----------------")
print("VQ_GENERATE_DONE")
