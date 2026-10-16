"""Vision front end: frozen CLIP features + the trainable projector."""

from .features import ClipPatchEncoder, load_features, preprocess, save_features

__all__ = ["ClipPatchEncoder", "load_features", "preprocess", "save_features"]
