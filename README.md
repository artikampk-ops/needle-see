# needle-see

Give Cactus Needle 3 eyes **without** giving it a VLM.

A standalone visual graft: frozen CLIP ViT-B/32 image features (int8 ONNX, in
cache on this laptop) are projected into Needle's embedding space and fed to the
stock Needle 3 JAX network as a *continuous visual prefix* in front of the byte
tokens.  Only a small projector + attention LoRA are trained on a synthetic, exact-
label, typed-decision suite, so — unlike a llava-class VLM — there is no
second, generative model in the vision path and no text generator emitting the
answer from captioning.

Stage status:

- Stage 0 (plumbing): ✓ project skeleton, env, forked `VisualSAN`, CLIP patch
  features, synthetic generator, smoke-tested end-to-end on CPU.
- Stage 1 (training): locally smoke-tested (2-layer rung, 96 rows, 8 epochs, 8.9
  min).  The *real* 20-layer run is the Kaggle notebook the user executes
  (`kaggle/run_kaggle.py`).
- Stage 2 (runtime): ✓ Jev-shaped CLI (`python -m needlesee.runtime`).
- Stage 3 (native ship): blocked upstream — see SPEC.md (also: `box` primitive).

## Quick start (on this laptop)

```bash
# 1. environment (already created at .venv; rebuild if needed)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python jax flax optax numpy safetensors \
    sentencepiece pillow onnxruntime onnx huggingface_hub pytest

# 2. expose patch tokens in the cached CLIP export (one-time)
.venv/bin/python tools/export_clip_patches.py

# 3. synthetic data
.venv/bin/python -m needlesee.data.gen --out out/data --count 4000 --seed 1 --split train
.venv/bin/python -m needlesee.data.gen --out out/data --count 500  --seed 2 --split val

# 4. frozen features
.venv/bin/python -m needlesee.data.precompute --jsonl out/data/train.jsonl --out out/features
.venv/bin/python -m needlesee.data.precompute --jsonl out/data/val.jsonl   --out out/features

# 5. train (smoke)
.venv/bin/python -m needlesee.train --smoke

# 6. ask one question
.venv/bin/python -m needlesee.runtime --bundle out/graft \
    --image out/data/val/images/000000.jpg \
    --choice "What color is the circle?" \
    --options red green blue yellow purple orange

# 7. tests
.venv/bin/python -m pytest -q
```

## Layout

```
needlesee/
  paths.py                 # vendored-package shim + cache paths
  model3.py                # VisualSAN: needle 3 SAN + continuous visual prefix
  projector.py             # CLIP patch -> Needle embedding projector
  graft.py                 # train/runtime artifact bundle
  questions.py             # data formats (typesafe of Jev)
  vision/                  # CLIP feature encoder (ONNX)
  data/gen.py              # synthetic exact-label suite
  data/precompute.py       # feature cache (N, 50, 768) fp16
  train/train_graft.py     # jax LoRA + projector training
  runtime/decoding.py       # candidate scoring + greedy generation
  runtime/agent.py          # GraftRuntime
  runtime/cli.py            # flags + Jev-shaped request JSON
tests/                       # equivalence + data + model tests
SPEC.md                     # the Jev contract captured
```

## Training results (measured locally, v1 smoke)

| rung | rows | epochs | CLIP ms/img | train min | candidate accuracy (val) |
|------|------|--------|-------------|-----------|--------------------------|
| 2    | 96   | 8      | ~60         | 8.9       | 46.9 % (32 ex, 8 tasks)  |

Baseline for comparison (trained model, *zero* image features instead of CLIP):
≈ 37.5 % on the same 8-example slice — i.e. the visual prefix carries a small but
measurable signal even at two layers.  Expect the 20-layer Kaggle run to show a much
larger gap; *claims are measured, not asserted.*

## Latency (this laptop, Intel i5-6400, no GPU)

- CLIP int8 vision: ~60 ms / image.
- Graft 2-layer decision (candidate scoring, 6-way): ~30–40 s warm, ~1–2 s JIT pause
  first call.  20-layer will be ~1–2 min on CPU; on Kaggle GPU via jax[cuda12]
  it should drop into the single-digit-second range, but measure there — numbers
  from this CPU do not travel to GPUs or other runs unearned.
- Greedy generation (long-form completion): ~1 s/step at 2 layers; candidate
  scoring is preferred for typed decisions (no auto-regression in the hot path).

## Notes
- The native engine/target format is text-only (`libneedle.so` provides
  `needle_init/needle_complete/needle_embed` over strings and `needle build` writes
  text archives).  This graft ships as a safetensors LoRA + projector bundle
  runnable by this project's runtime, not the `.cact` engine.
- `box` is intentionally deferred; candidate labels include positions only in the
  synthetic suite later. Confidence reports `uncalibrated: true`.
- Confidence head is frozen (attention/projector only) in v1.
