"""The graft bundle: what training writes and the runtime reads back.

A bundle directory contains

* ``lora.safetensors``      - LoRA adapters for the base checkpoint (vendored format)
* ``projector.safetensors`` - the visual projector parameters (flax-flat names)
* ``graft.json``            - everything needed to rebuild the model
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import ensure_needle
from .projector import VisualProjector

ensure_needle()

from flax.traverse_util import flatten_dict, unflatten_dict  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402

from needle.model.checkpoints import read_adapter, write_adapter  # noqa: E402

FORMAT = "needle-see/graft/v1"


@dataclass
class GraftBundle:
    meta: dict
    lora: dict
    scale: float
    projector: VisualProjector
    projector_params: dict

    @property
    def base(self) -> str:
        return self.meta["base"]

    @property
    def layers(self) -> int:
        return int(self.meta["layers"])


def save_bundle(out_dir: str | Path, *, lora, scale, rank, seed, base, layers,
                projector_params, projector, stats: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_adapter(out_dir / "lora.safetensors", {
        "lora": {"/".join(path): {"A": np.asarray(value["A"]),
                                  "B": np.asarray(value["B"])}
                 for path, value in lora.items()},
        "scale": float(scale),
        "base": str(base),
        "rank": int(rank),
        "seed": int(seed),
    })
    flat = {"/".join(key): np.asarray(value)
            for key, value in flatten_dict(projector_params).items()}
    save_file(flat, str(out_dir / "projector.safetensors"))
    meta = {
        "format": FORMAT,
        "base": str(base),
        "layers": int(layers),
        "lora_rank": int(rank),
        "lora_scale": float(scale),
        "seed": int(seed),
        "feature_dim": int(projector.feature_dim),
        "feature_tokens": int(projector.num_tokens),
        "projector_hidden": int(projector.hidden),
        "d_model": int(projector.d_model),
        **stats,
    }
    (out_dir / "graft.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_bundle(out_dir: str | Path) -> GraftBundle:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "graft.json").read_text())
    adapter = read_adapter(out_dir / "lora.safetensors")
    lora = {tuple(name.split("/")): {"A": np.asarray(value["A"]),
                                     "B": np.asarray(value["B"])}
            for name, value in adapter["lora"].items()}
    flat = load_file(str(out_dir / "projector.safetensors"))
    projector_params = unflatten_dict({
        tuple(key.split("/")): np.asarray(value) for key, value in flat.items()
    })
    projector = VisualProjector(d_model=meta["d_model"],
                                num_tokens=meta["feature_tokens"],
                                feature_dim=meta["feature_dim"],
                                hidden=meta["projector_hidden"])
    return GraftBundle(meta=meta, lora=lora, scale=float(meta["lora_scale"]),
                       projector=projector, projector_params=projector_params)
