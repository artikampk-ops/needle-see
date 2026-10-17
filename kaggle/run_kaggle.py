"""Kaggle entry point for the 20-layer graft.

Usage:
    python kaggle/run_kaggle.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path("/kaggle/working/needle-see")
if not PROJECT.exists():
    PROJECT = Path.cwd()
    if not (PROJECT / "needlesee").exists():
        print("ERROR: project not found", flush=True)
        sys.exit(1)

os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
os.environ["PYTHONPATH"] = str(PROJECT) + os.pathsep + os.environ.get("PYTHONPATH", "")

print(f"PROJECT={PROJECT}", flush=True)
print(f"cwd={os.getcwd()}", flush=True)
print(f"needlesee exists: {(PROJECT / 'needlesee').exists()}", flush=True)
print(f"needlesee/data exists: {(PROJECT / 'needlesee' / 'data').exists()}", flush=True)
print(f"needlesee/data/gen.py exists: {(PROJECT / 'needlesee' / 'data' / 'gen.py').exists()}", flush=True)

COUNT = int(os.environ.get("NEEDLE_COUNT", "24000"))
VAL = int(os.environ.get("NEEDLE_VAL", "2400"))
DATA = "data"
FEATURES = "features"
OUT = "graft"


# Ensure the patched CLIP model is available
clip_onnx = PROJECT / "models" / "clip_vision_patches.onnx"
if not clip_onnx.exists():
    print("Downloading Xenova CLIP model and exporting patch tokens...", flush=True)
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("Xenova/clip-vit-base-patch32",
                          "onnx/vision_model_quantized.onnx")
    print(f"  downloaded to {src}", flush=True)
    # Run exporter
    import subprocess
    result = subprocess.run(
        [sys.executable, "tools/export_clip_patches.py", "--source", str(src),
         "--out", str(clip_onnx)],
        cwd=str(PROJECT), env=os.environ.copy())
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"  exported to {clip_onnx}", flush=True)
else:
    print(f"CLIP model already present: {clip_onnx}", flush=True)


# Generate data if needed
if not Path(f"{DATA}/train.jsonl").exists():
    print("Generating training data...", flush=True)
    from needlesee.data.gen import generate
    generate(DATA, COUNT, 1, "train", noise="light", rotate=4)
    generate(DATA, VAL, 2, "val", noise="light", rotate=4)
    print("Data generation complete.", flush=True)
else:
    print("Training data already exists, skipping generation.", flush=True)

# Precompute features if needed
if not Path(f"{FEATURES}/train.npy").exists():
    print("Precomputing features...", flush=True)
    from needlesee.data.precompute import precompute
    precompute(f"{DATA}/train.jsonl", FEATURES)
    precompute(f"{DATA}/val.jsonl", FEATURES)
    print("Feature precomputation complete.", flush=True)
else:
    print("Features already exist, skipping precomputation.", flush=True)

# Train
import sys as _sys
_sys.argv = [
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
    _sys.argv.append("--smoke")

from needlesee.train.train_graft import main as train_main
code = train_main()
print("train() returned", code, flush=True)

# Print the graft.json for easy extraction from kernel log
import json as _json
graft_json_path = PROJECT / OUT / "graft.json"
if graft_json_path.exists():
    print("GRAFT_JSON_START", flush=True)
    print(graft_json_path.read_text(), flush=True)
    print("GRAFT_JSON_END", flush=True)

# Clean up large intermediate files to reduce output size
import shutil
for d in [DATA, FEATURES]:
    dpath = PROJECT / d
    if dpath.exists():
        shutil.rmtree(dpath)
        print(f"Cleaned up {d}/", flush=True)
