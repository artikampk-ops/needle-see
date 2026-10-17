"""Kaggle entry point for the 20-layer graft (run after uploading the project).

Usage:
    python kaggle/run_kaggle.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/kaggle/working/needle-see")
if PROJECT.exists():
    os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

COUNT = int(os.environ.get("NEEDLE_COUNT", "24000"))
VAL = int(os.environ.get("NEEDLE_VAL", "2400"))
DATA = "data"
FEATURES = "features"
OUT = "graft"


def run(*argv):
    print(" ".join(argv), flush=True)
    result = subprocess.run(list(argv))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if not Path(f"{DATA}/train.jsonl").exists():
    run("python", "-m", "needlesee.data.gen", "--out", DATA,
        "--count", str(COUNT), "--seed", "1", "--split", "train",
        "--noise", "light", "--rotate", "4")
    run("python", "-m", "needlesee.data.gen", "--out", DATA,
        "--count", str(VAL), "--seed", "2", "--split", "val",
        "--noise", "light", "--rotate", "4")

if not Path(f"{FEATURES}/train.npy").exists():
    run("python", "-m", "needlesee.data.precompute", "--jsonl",
        f"{DATA}/train.jsonl", "--out", FEATURES)
    run("python", "-m", "needlesee.data.precompute", "--jsonl",
        f"{DATA}/val.jsonl", "--out", FEATURES)

sys.argv = [
    "train", "--data-dir", DATA, "--features", FEATURES, "--out", OUT,
    "--layers", os.environ.get("NEEDLE_LAYERS", "20"),
    "--epochs", os.environ.get("NEEDLE_EPOCHS", "6"),
    "--batch-size", os.environ.get("NEEDLE_BATCH", "32"),
    "--lora-rank", os.environ.get("NEEDLE_RANK", "16"),
    "--lora-alpha", os.environ.get("NEEDLE_ALPHA", "32"),
    "--eval-val", os.environ.get("NEEDLE_EVAL", "512"),
    "--seed", "0",
]
if os.environ.get("NEEDLE_SMOKE") == "1":
    sys.argv.append("--smoke")

from needlesee.train.train_graft import main as train_main
code = train_main()
print("train() returned", code)
