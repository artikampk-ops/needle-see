"""Path helpers and the vendored-needle shim.

The project works against either a pip-installed ``cactus-needle`` or the
vendored checkout in ``~/Work/system1-research/vendor/needle``.  ``ensure_needle``
makes ``import needle`` resolve either way before any needle import runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

# ~/Work/system1-research/needle-see -> ~/Work/system1-research/vendor/needle
DEFAULT_VENDOR = PROJECT.parent / "vendor" / "needle"
DEFAULT_CLIP_DIR = Path(os.path.expanduser("~/.cache/hf/Xenova/clip-vit-base-patch32"))

HF_NEEDLE_REPO = "Cactus-Compute/needle3"
HF_CHECKPOINT = "checkpoints/needle3.safetensors"

_NEEDLE_READY = False


def ensure_needle() -> None:
    """Make ``import needle`` work, from the environment or the vendored checkout."""
    global _NEEDLE_READY
    if _NEEDLE_READY:
        return
    try:
        import needle  # noqa: F401
    except ImportError:
        src = Path(os.environ.get("NEEDLE_SRC", DEFAULT_VENDOR)).resolve()
        if (src / "needle" / "__init__.py").is_file():
            sys.path.insert(0, str(src))
            import needle  # noqa: F401
        else:
            raise ImportError(
                "cactus-needle is not importable and no checkout exists at "
                f"{src}; `pip install cactus-needle` or set NEEDLE_SRC."
            ) from None
    _NEEDLE_READY = True


def checkpoint_path() -> Path:
    """Resolve ``checkpoints/needle3.safetensors``: env, HF cache, then download."""
    override = os.environ.get("NEEDLE3_CKPT")
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"NEEDLE3_CKPT points nowhere: {path}")
        return path
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    pattern = f"models--Cactus-Compute--needle3/snapshots/*/{HF_CHECKPOINT}"
    cached = sorted(p for p in hub.glob(pattern) if p.is_file())
    if cached:
        return cached[-1]
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_NEEDLE_REPO, HF_CHECKPOINT))


def clip_model_dir() -> Path:
    """Directory holding ``onnx/vision_model_quantized.onnx`` and preprocessor config."""
    override = os.environ.get("CLIP_DIR")
    directory = Path(override) if override else DEFAULT_CLIP_DIR
    onnx = directory / "onnx" / "vision_model_quantized.onnx"
    if not onnx.is_file():
        raise FileNotFoundError(
            f"quantized CLIP vision model not found at {onnx}; "
            "set CLIP_DIR to a directory holding onnx/vision_model_quantized.onnx"
        )
    return directory


def clip_patch_model() -> Path:
    """The CLIP vision ONNX with patch tokens exposed (see tools/export_clip_patches.py)."""
    override = os.environ.get("CLIP_PATCH_ONNX")
    candidates = [Path(override)] if override else []
    candidates.append(PROJECT / "models" / "clip_vision_patches.onnx")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "clip patch-token ONNX not found; run tools/export_clip_patches.py "
        f"(looked at {', '.join(str(c) for c in candidates)})"
    )
