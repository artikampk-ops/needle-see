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
if not PROJECT.exists():
    # Maybe cloned elsewhere; try cwd
    PROJECT = Path.cwd()
    if not (PROJECT / "needlesee").exists():
        print("ERROR: project not found at /kaggle/working/needle-see or cwd", flush=True)
        sys.exit(1)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ["PYTHONPATH"] = str(PROJECT) + os.pathsep + os.environ.get("PYTHONPATH", "")

print(f"PROJECT={PROJECT}", flush=True)
print(f"cwd={os.getcwd()}", flush=True)
print(f"needlesee exists: {(PROJECT / 'needlesee').exists()}", flush=True)

# Ensure the patched CLIP model is available (download Xenova if needed)
clip_onnx = PROJECT / "models" / "clip_vision_patches.onnx"
if not clip_onnx.exists():
    print("Downloading Xenova CLIP model and exporting patch tokens...", flush=True)
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("Xenova/clip-vit-base-patch32",
                          "onnx/vision_model_quantized.onnx")
    print(f"  downloaded to {src}", flush=True)
    run("python", "tools/export_clip_patches.py", "--source", str(src),
        "--out", str(clip_onnx))
    print(f"  exported to {clip_onnx}", flush=True)

COUNT = int(os.environ.get("NEEDLE_COUNT", "24000"))
VAL = int(os.environ.get("NEEDLE_VAL", "2400"))
DATA = "data"
FEATURES = "features"
OUT = "graft"


def run(*argv):
    print(" ".join(argv), flush=True)
    result = subprocess.run(list(argv), cwd=str(PROJECT), env=os.environ.copy())
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
