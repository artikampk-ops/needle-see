# Jev "see" (v1) feature spec as implemented by `needle-see`
Status: v0.1 — noul / choice / score only. Box is deferred to v2.

## Where this runs
`~/Work/system1-research/needle-see/` is a standalone project (not a patch to Cactus
Needle).  It deliberately keeps the vendored Needle 3 source untouched and imports it
via `needlesee.paths.ensure_needle()`.

## Jev contract captured from docs.typesafe.ai (go/images, primitives/box, /primitives/noul)

### Request surface
- Field `assets`: object keyed by reference name; each asset is
  `{"type": "image", "format": "png"|"jpg"|"jpeg", "data_base64": ...}` or a multipart
  `part` (a path-like reference in the local CLI).
- Reference strings `{{assetName}}` may appear *only as values* inside `state`,
  `instructions`, `criteria`, `examples`, `tools`/`tools_arguments`.  Whole-value
  substitution first, then left-to-right, non-overlapping, longest match: a
  reference planted inside another reference is not carried through; object *keys*
  are never replaced; referencing an undeclared asset is a 400.
- Image gated by header `TypeSafe-Danger-Zone-Images: true` (the local runtime does
  not enforce the header; it is documented for proxying against a real Jev API).
- Limits: side 280–1568 px, 16 MiB/image, 32 MiB/request, formats as above.

### Primitive results
- `noul`  -> `{"noul": 0..1, "calibrated": bool, ...}` (probability)
- `choice` -> `{"value": label, "distribution": {label: p, ...}, "confidence": p, "calibrated": bool}`
- `score`  -> `{"value": real, "distribution": {level: p, ...}, "confidence": p, "calibrated": bool}`
- `box`   -> `{top, bottom, left, right} fractions + confidence` — **deferred to v2** in
  this project (the one thing the local graft does not yet do).

## Local design (the “true graft”)
- No VLM. The visual information flows through Needle 3’s own weights.
- Frozen CLIP ViT-B/32 (`Xenova/clip-vit-base-patch32`, int8 ONNX, in cache) →
  *last hidden state* (50 × 768, exposed locally by `tools/export_clip_patches.py`)
  → a small trainable two-layer projector → prepended visual-token prefix into a
  forked `needlesee.model3.VisualSAN`.
- The fork adds an `inputs_embeds`-style *continuous prefix* (V visual vectors in
  front of the byte tokens); the engram byte memory is masked so no window that
  intersects the prefix is ever read; the prefix is attention-visible but excluded
  from the probe heads (confidence/embedding/router) in v1.
- Training is JAX LoRA on the attention projections (rank 8–16) + the projector,
  with the same masked cross-entropy and prompt wire format the stock Needle local
  fine-tune uses, on a *synthetic, exact-label* suite (shapes, dialogs, receipts;
  12 task types, noul/choice/rate).
- Base checkpoint `Cactus-Compute/needle3` (121 M, 20 layers) → rung-sliced to 2/4/8
  for Kaggle budgets; the full 20-layer run is the Kaggle target.

## Known limitations (be honest)
- On the SQLite-CPU machine this runs on, a single multi-answer decision takes
  roughly 30 s at 2 layers and ~1–2 min at 20 layers; CLIP int8 ≈ 60 ms/image.
  Latency is measured, not assumed — see the benchmark notes in README.md.
- The graft cannot ship through `needle build` to a `.cact`: the native engine
  (`libneedle.so`) accepts strings only (`needle_init/needle_complete/needle_embed`
  are text-in), and the engine reads text-only archives.  A future Stage 3 would need
  upstream Cactus changes to accept visual prefixes.  Training output is a
  `.safetensors` LoRA + projector bundle.
- `box` (fractions) is deferred to v2 (needs spatial labels).
- Confidence is reported with `uncalibrated: true` — the confidence head stays
  frozen (only attention + projector train), so it is not calibrated for mixed
  vision/text prompts in v1.
- Receipt reading is known to be weak: CLIP ViT-B/32 at 224 px erases small text;
  include the task as an honest probe.
