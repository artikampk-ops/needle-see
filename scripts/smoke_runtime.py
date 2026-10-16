"""Smoke test the saved graft end-to-end: load bundle -> candidate score + generate."""

import os, json
import numpy as np
import jax, jax.numpy as jnp
from PIL import Image

from needlesee.paths import ensure_needle, checkpoint_path; ensure_needle()
from needlesee.graft import load_bundle
from needlesee.model3 import VisualSAN
from needlesee.projector import VisualProjector
from needlesee.runtime.decoding import (
    score_completions, softmax, greedy_generate, parse_calls,
)
from needle.model.run import load_checkpoint
from needle.model.finetune import merge_lora, rung, render_example
from needle.model.tokenizer import BOS_ID

base_dir = os.path.abspath("out/graft")
bundle = load_bundle(base_dir)
params, config = load_checkpoint(str(checkpoint_path()))
config.dtype = "float32"
params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
params = jax.device_put(params)
params, config = rung(params, config, bundle.layers)
params = jax.device_put(params)

model = VisualSAN(config)
merged = merge_lora(params, bundle.lora, bundle.scale)
projector_params = jax.device_put(bundle.projector_params)

# Pick one val example
rows = [json.loads(l) for l in open("out/data/val.jsonl")]
row = rows[0]
features = np.load("out/features/val.npy", mmap_mode="r")[0]
image_path = os.path.join("out/data", row["image"])
print("task:", row["task"], "| truth:", row["answers"], "| image:", row["image"])

# Candidate scoring
prompt, _ = render_example({**row, "answers": []})
prompt_ids = [BOS_ID] + __import__("needlesee.runtime.decoding", fromlist=["_"]).tokenizer.encode(prompt) if False else None
from needle.model.tokenizer import get_tokenizer
tokenizer = get_tokenizer(config.vocab_size)
prompt_ids = [BOS_ID] + tokenizer.encode(prompt)
scores = score_completions(model, merged, bundle.projector, projector_params,
                           features, tokenizer, prompt_ids, row["candidates"])
probs = softmax(scores)
print("candidates:", row["candidates"])
print("scores:", np.round(scores, 3).tolist())
print("probs :", np.round(probs, 3).tolist())
print("pred idx:", int(np.argmax(probs)), "truth idx:", int(row["truth_index"]))

# Greedy generation
answer = greedy_generate(model, merged, bundle.projector, projector_params,
                         np.asarray(features), tokenizer, prompt_ids,
                         max_new_tokens=64)
print("generated:", repr(answer))
print("parsed calls:", parse_calls(answer))
