"""CLI for the grafted Needle 3 vision runtime.

Examples
  python -m needlesee.runtime --bundle out/graft --image img.jpg --choice "What color is the circle?" red green blue
  python -m needlesee.runtime --bundle out/graft --image img.jpg --noul "Is there a red circle?"
  python -m needlesee.runtime --bundle out/graft --image img.jpg --score "How busy?" none one two three-or-more
  python -m needlesee.runtime --bundle out/graft --request request.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..paths import ensure_needle

ensure_needle()

from ..questions import ChoiceQuestion, JudgeQuestion, RateQuestion
from .agent import GraftRuntime

ensure_needle()


def _load_image_asset(asset):
    """Load a Jev-style image asset into path-or-feature form (path preferred).

    Accepts: {"part": path}, {"data_base64": b64}, {"path": path},
    or a bare string path / data: URI.
    """
    from PIL import Image
    from io import BytesIO
    import base64
    if isinstance(asset, str):
        if asset.startswith("data:"):
            encoded = asset.split(",", 1)[1]
            return BytesIO(base64.b64decode(encoded))
        return Image.open(asset).convert("RGB")
    if hasattr(asset, "read"):
        return Image.open(asset).convert("RGB")
    if isinstance(asset, dict):
        if "part" in asset:
            return Image.open(asset["part"]).convert("RGB")
        if "path" in asset:
            return Image.open(asset["path"]).convert("RGB")
        if "data_base64" in asset:
            return Image.open(BytesIO(base64.b64decode(asset["data_base64"]))).convert("RGB")
        if "url" in asset:
            from urllib.request import urlopen
            return Image.open(BytesIO(urlopen(asset["url"]).read())).convert("RGB")
    raise ValueError(f"unrecognised image asset: {asset!r}")


def _question_from_jev(qid, spec):
    """Convert a Jev-shaped question spec into a Question."""
    qtype = spec.get("type", "choice")
    text = spec.get("text") or spec.get("question") or spec.get("query") or ""
    if qtype in ("noul", "judge", "yesno"):
        return JudgeQuestion(text)
    if qtype in ("choice", "pick"):
        return ChoiceQuestion(text, tuple(spec.get("options") or []))
    if qtype in ("score", "rate"):
        return RateQuestion(text, tuple(spec.get("levels") or spec.get("options") or ()))
    # box is deferred to a later stage.
    raise ValueError(f"question '{qid}': unsupported type {qtype!r} (box is not in v1)")


def run_flag(args):
    runtime = GraftRuntime(args.bundle, dtype=args.dtype)
    images = args.image or []
    if not images:
        raise SystemExit("--image is required in flag mode")
    image_ref = images[0]
    answers = {}
    if args.noul:
        answers["noul"] = runtime.ask(image_ref, JudgeQuestion(args.noul))
    if args.choice:
        if not args.options:
            raise SystemExit("--options required with --choice")
        answers["choice"] = runtime.ask(image_ref, ChoiceQuestion(args.choice, tuple(args.options)))
    if args.score:
        if not args.levels:
            raise SystemExit("--levels required with --score")
        answers["score"] = runtime.ask(image_ref, RateQuestion(args.score, tuple(args.levels)))
    if not answers:
        raise SystemExit("pass --noul / --choice / --score")
    print(json.dumps({"image": image_ref, "answers": answers}, indent=2, ensure_ascii=False))


def run_request(args):
    runtime = GraftRuntime(args.bundle, dtype=args.dtype)
    request = json.loads(Path(args.request).read_text())
    assets = request.get("assets", {})
    questions = request.get("questions", {})
    images = [_load_image_asset(assets[k]) for k in assets]
    image_ref = images[0] if images else None
    rendered = {}
    for qid, spec in questions.items():
        question = _question_from_jev(qid, spec)
        rendered[qid] = runtime.ask(image_ref, question)
    response = {
        "model": "needle-see-3-graft",
        "answers": rendered,
        "usage": {"image_inputs": len(images)},
        "note": "confidence is uncalibrated (v1)",
    }
    print(json.dumps(response, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="needle-see vision runtime")
    parser.add_argument("--bundle", required=True,
                        help="directory produced by the graft trainer")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--request", default=None, help="Jev-shaped request JSON")
    parser.add_argument("--image", action="append", default=[],
                        help="path (or repeated) for flag mode")
    parser.add_argument("--noul", default=None, help="yes/no question")
    parser.add_argument("--choice", default=None, help="choice/multiple choice question")
    parser.add_argument("--options", nargs="+", default=None)
    parser.add_argument("--score", default=None, help="score/level question")
    parser.add_argument("--levels", nargs="+", default=None)
    args = parser.parse_args()
    if args.request:
        run_request(args)
    else:
        run_flag(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())