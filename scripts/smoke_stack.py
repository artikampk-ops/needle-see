"""Scratch smoke test: checkpoint + tokenizer + forward + generate on CPU.

Run:  .venv/bin/python scripts/smoke_stack.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "needle"))

import jax
import jax.numpy as jnp
import numpy as np

from needle.model.run import load_checkpoint, generate
from needle.model.tokenizer import get_tokenizer

CKPT = "/home/maaz/.cache/huggingface/hub/models--Cactus-Compute--needle3/snapshots/93e88ede867e4311c39151285e74df513e4b835b/checkpoints/needle3.safetensors"

t0 = time.time()
params, config = load_checkpoint(CKPT)
print(f"checkpoint loaded in {time.time()-t0:.1f}s")
print("config:", config)
print("param tree keys:", list(params.keys()))
n_params = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
print(f"params: {n_params/1e6:.1f}M, dtype set: {set(str(v.dtype) for v in jax.tree_util.tree_leaves(params)[:5])}")

t0 = time.time()
tok = get_tokenizer()
print(f"tokenizer ready in {time.time()-t0:.1f}s, vocab {tok.vocab_size}")
ids = tok.encode("hello world")
print("encode ->", ids[:12], "decode ->", repr(tok.decode(ids)[:40]))

from needle.model.architecture import SimpleAttentionNetwork

model = SimpleAttentionNetwork(config)


@jax.jit
def forward(params, ids, mask):
    return model.apply({"params": params}, ids, mask, quant=False)


rng = np.random.default_rng(0)
tokens = jnp.asarray([tok.encode("hello world")])
t0 = time.time()
logits = forward(params, tokens, None)
print(f"jit+forward done in {time.time()-t0:.1f}s; logits {logits.shape} {logits.dtype}")
nxt = int(jnp.argmax(logits[0, -1]))
print("next token argmax:", nxt, repr(tok.decode([nxt])))
