"""Expose CLIP patch tokens from the cached Xenova ONNX vision tower.

The ``Xenova/clip-vit-base-patch32`` vision export only surfaces the pooled
``image_embeds``.  The transformer's last hidden state — ``(batch, 50, 768)``
patch tokens — is computed internally and then pooled.  This tool loads the
graph, adds that tensor as an output (no math changes), and saves a copy.

Usage:
    .venv/bin/python tools/export_clip_patches.py [--out models/clip_vision_patches.onnx]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

LAST_HIDDEN = "/vision_model/encoder/layers.11/Add_1_output_0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="vision_model_quantized.onnx (default: HF cache)")
    ap.add_argument("--out", default="models/clip_vision_patches.onnx")
    ap.add_argument("--tokens", type=int, default=50, help="1 + patch grid count")
    ap.add_argument("--hidden", type=int, default=768)
    args = ap.parse_args()

    source = args.source
    if source is None:
        source = os.path.expanduser(
            "~/.cache/hf/Xenova/clip-vit-base-patch32/onnx/vision_model_quantized.onnx")
    if not os.path.isfile(source):
        print(f"source not found: {source}", file=sys.stderr)
        return 1

    model = onnx.load(source)
    graph = model.graph
    names = {output.name for output in graph.output}
    if LAST_HIDDEN not in names:
        graph.output.append(helper.make_tensor_value_info(
            LAST_HIDDEN, TensorProto.FLOAT,
            ["batch_size", args.tokens, args.hidden]))
    # Keep the original pooled output too, under a stable name we know.
    onnx.checker.check_model(model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("outputs:", [o.name for o in graph.output])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
