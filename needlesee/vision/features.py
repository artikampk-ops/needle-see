"""Frozen CLIP vision features for the graft.

Encoder: the cached int8 ``clip-vit-base-patch32`` vision tower, with the last
hidden state exposed as an extra output (see ``tools/export_clip_patches.py``).
Features are the 50 patch-grid tokens (1 CLS + 49 patches) of shape 768, which
the trainable projector maps into Needle's embedding space.  No generative
model is involved anywhere in this path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..paths import clip_patch_model

# CLIPImageProcessor defaults for clip-vit-base-patch32.
MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.asarray([0.26862954, 0.26130258, 0.27577711], np.float32)
SIZE = 224
PATCHES_OUTPUT = "patch_tokens"


def preprocess(image: Image.Image, size: int = SIZE) -> np.ndarray:
    """Resize shortest edge to ``size``, center crop, normalize. (3, size, size)."""
    image = image.convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    if scale != 1:
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.BICUBIC)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))


class ClipPatchEncoder:
    """onnxruntime-backed CLIP vision encoder returning patch tokens."""

    def __init__(self, model_path: str | Path | None = None,
                 intra_op_threads: int | None = None,
                 include_pooled: bool = False):
        import onnxruntime as ort

        path = Path(model_path) if model_path else clip_patch_model()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        names = [o.name for o in self.session.get_outputs()]
        if len(names) < 2:
            raise RuntimeError(
                f"{path} has no patch-token output ({names}); run "
                "tools/export_clip_patches.py first")
        self._patch_output = names[1]
        self._pooled_output = names[0]
        self.include_pooled = include_pooled

    def encode(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Encode PIL images -> (N, 50, 768) float32 patch tokens."""
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            pixel_values = np.stack([preprocess(img) for img in batch])
            outputs = self.session.run([self._patch_output],
                                       {"pixel_values": pixel_values})
            chunks.append(np.asarray(outputs[0], np.float32))
        if not chunks:
            return np.zeros((0, 50, 768), np.float32)
        return np.concatenate(chunks, axis=0)

    def encode_files(self, paths: list[str | Path], load_pooled: bool = False,
                     batch_size: int = 16):
        """Encode image files; optionally also return pooled (N, 512) embeds."""
        images = [Image.open(p).convert("RGB") for p in paths]
        if not load_pooled:
            return self.encode(images, batch_size=batch_size)
        patches, pooled = [], []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            pixel_values = np.stack([preprocess(img) for img in batch])
            out = self.session.run([self._patch_output, self._pooled_output],
                                   {"pixel_values": pixel_values})
            patches.append(np.asarray(out[0], np.float32))
            pooled.append(np.asarray(out[1], np.float32))
        return np.concatenate(patches), np.concatenate(pooled)


def save_features(path: str | Path, features: np.ndarray, dtype=np.float16) -> None:
    np.save(path, features.astype(dtype))


def load_features(path: str | Path, mmap: bool = True) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)
