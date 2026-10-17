"""Kaggle entry point for the 20-layer graft.

Usage:
    python kaggle/run_kaggle.py
"""

from __future__ import annotations

import json
import os
import sys
import importlib.util
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

# Debug: list what's actually in needlesee/
print(f"PROJECT={PROJECT}", flush=True)
print(f"cwd={os.getcwd()}", flush=True)
needlesee_dir = PROJECT / "needlesee"
print(f"needlesee exists: {needlesee_dir.exists()}", flush=True)
if needlesee_dir.exists():
    print(f"needlesee contents: {sorted(p.name for p in needlesee_dir.iterdir())}", flush=True)
    data_dir = needlesee_dir / "data"
    print(f"needlesee/data exists: {data_dir.exists()}", flush=True)
    if data_dir.exists():
        print(f"needlesee/data contents: {sorted(p.name for p in data_dir.iterdir())}", flush=True)

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


# Helper to import needlesee submodules robustly
def _import_module(relpath, modname):
    """Import a module from a relative path, bypassing package __init__ issues."""
    full = PROJECT / relpath
    if not full.exists():
        raise FileNotFoundError(f"{full} not found")
    spec = importlib.util.spec_from_file_location(modname, full)
    mod = importlib.util.module_from_spec(spec)
    # Ensure parent packages are in sys.modules
    parts = modname.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            parent_spec = importlib.util.spec_from_file_location(
                parent_name, PROJECT / "/".join(parts[:i]) / "__init__.py")
            if parent_spec and parent_spec.loader:
                parent_mod = importlib.util.module_from_spec(parent_spec)
                sys.modules[parent_name] = parent_mod
                parent_spec.loader.exec_module(parent_mod)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# Import needlesee (ensure __init__.py files exist)
for pkg in ["needlesee", "needlesee.data", "needlesee.train", "needlesee.runtime",
            "needlesee.vision"]:
    init_path = PROJECT / pkg.replace(".", "/") / "__init__.py"
    if not init_path.exists():
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text("")
    if pkg not in sys.modules:
        _import_module(pkg.replace(".", "/") + "/__init__.py", pkg)

# Now import the needed functions
gen_mod = _import_module("needlesee/data/gen.py", "needlesee.data.gen")
precompute_mod = _import_module("needlesee/data/precompute.py", "needlesee.data.precompute")
train_mod = _import_module("needlesee/train/train_graft.py", "needlesee.train.train_graft")

generate = gen_mod.generate
precompute = precompute_mod.precompute
train_main = train_mod.main

# Generate data if needed
if not Path(f"{DATA}/train.jsonl").exists():
    print("Generating training data...", flush=True)
    generate(DATA, COUNT, 1, "train", noise="light", rotate=4)
    generate(DATA, VAL, 2, "val", noise="light", rotate=4)
    print("Data generation complete.", flush=True)
else:
    print("Training data already exists, skipping generation.", flush=True)

# Precompute features if needed
if not Path(f"{FEATURES}/train.npy").exists():
    print("Precomputing features...", flush=True)
    precompute(f"{DATA}/train.jsonl", FEATURES)
    precompute(f"{DATA}/val.jsonl", FEATURES)
    print("Feature precomputation complete.", flush=True)
else:
    print("Features already exist, skipping precomputation.", flush=True)

# Train
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

code = train_main()
print("train() returned", code, flush=True)

# Print the graft.json for easy extraction from kernel log
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
