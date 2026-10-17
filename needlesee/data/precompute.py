"""Precompute frozen CLIP patch features for a generated split.

    .venv/bin/python -m needlesee.data.precompute --jsonl out/data/train.jsonl \
        --out out/features

Writes ``<out>/<split>.npy`` (N, 50, 768, float16, row == jsonl line order).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..paths import ensure_needle
from ..vision.features import ClipPatchEncoder


def read_split(jsonl_path: str | Path):
    jsonl_path = Path(jsonl_path)
    base = jsonl_path.parent
    rows = []
    with jsonl_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return base, rows


def precompute(jsonl_path: str | Path, out_dir: str | Path, batch_size: int = 16,
               intra_threads: int | None = None) -> Path:
    ensure_needle()
    from PIL import Image

    base, rows = read_split(jsonl_path)
    encoder = ClipPatchEncoder(intra_op_threads=intra_threads)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(jsonl_path).stem
    chunks = []
    started = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = [Image.open(base / row["image"]).convert("RGB") for row in batch]
        chunks.append(encoder.encode(images, batch_size=batch_size))
        done = min(start + batch_size, len(rows))
        rate = done / max(time.time() - started, 1e-6)
        print(f"\r  encoded {done}/{len(rows)}  ({rate:.1f}/s)", end="", flush=True)
    print()
    features = np.concatenate(chunks, axis=0).astype(np.float16)
    path = out_dir / f"{name}.npy"
    np.save(path, features)
    ids_path = out_dir / f"{name}.ids.json"
    ids_path.write_text(json.dumps([row["id"] for row in rows]))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="precompute CLIP patch features")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", default="out/features")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    path = precompute(args.jsonl, args.out, args.batch_size, args.threads)
    size = path.stat().st_size / 1e6
    print(f"  wrote {path}  {size:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
