"""Runtime: ask the graft questions about an image (Jev-shaped I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from ..paths import checkpoint_path, ensure_needle
from ..graft import load_bundle
from ..model3 import VisualSAN
from ..questions import ChoiceQuestion, JudgeQuestion, RateQuestion, SYSTEM, TOOLS
from ..runtime.decoding import score_completions
from ..vision.features import ClipPatchEncoder

ensure_needle()

from needle.model.finetune import merge_lora, render_example, rung  # noqa: E402
from needle.model.run import load_checkpoint  # noqa: E402
from needle.model.tokenizer import BOS_ID, get_tokenizer  # noqa: E402


class GraftRuntime:
    def __init__(self, bundle_dir: str, dtype: str = "float32"):
        bundle = load_bundle(bundle_dir)
        self.bundle = bundle
        self.encoder = ClipPatchEncoder()
        params, config = load_checkpoint(str(checkpoint_path()))
        config.dtype = dtype
        params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
        params = jax.device_put(params)
        params, config = rung(params, config, bundle.layers)
        params = jax.device_put(params)
        merged = jax.device_put(merge_lora(params, bundle.lora, bundle.scale))
        self.params = params
        self.merged = merged
        self.config = config
        self.model = VisualSAN(config)
        self.projector = bundle.projector
        self.projector_params = jax.device_put(bundle.projector_params)
        self.tokenizer = get_tokenizer(config.vocab_size)
        self.dtype = dtype

    def encode_image(self, image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        return self.encoder.encode([image.convert("RGB")], batch_size=1)[0]

    def ask(self, image, question) -> dict:
        features = self.encode_image(image)
        prompt, _ = render_example({"system": SYSTEM, "query": question.query(),
                                     "tools": TOOLS, "answers": []})
        prompt_ids = [BOS_ID] + self.tokenizer.encode(prompt)
        scores = score_completions(self.model, self.merged, self.projector,
                                  self.projector_params, features, self.tokenizer,
                                  prompt_ids, question.candidate_texts())
        probs = softmax(scores)
        if isinstance(question, ChoiceQuestion):
            order = int(np.argmax(probs))
            return {
                "type": "choice",
                "choice": question.options[order],
                "options": list(question.options),
                "distribution": {opt: float(p) for opt, p in zip(question.options, probs)},
                "confidence": float(probs[order]),
                "uncalibrated": True,
            }
        if isinstance(question, JudgeQuestion):
            p_yes = float(probs[0])
            return {
                "type": "noul",
                "noul": p_yes,
                "answer": bool(p_yes >= 0.5),
                "distribution": {"true": p_yes, "false": 1.0 - p_yes},
                "confidence": max(p_yes, 1.0 - p_yes),
                "uncalibrated": True,
            }
        if isinstance(question, RateQuestion):
            order = int(np.argmax(probs))
            return {
                "type": "score",
                "score": order,
                "level": question.levels[order],
                "levels": list(question.levels),
                "distribution": {lvl: float(p) for lvl, p in zip(question.levels, probs)},
                "value": float(sum(i * p for i, p in enumerate(probs))),
                "confidence": float(probs[order]),
                "uncalibrated": True,
            }
        raise TypeError(f"unsupported question: {type(question)!r}")


def softmax(scores):
    scores = np.asarray(scores, np.float64)
    scores -= scores.max()
    exp = np.exp(scores)
    return exp / exp.sum()


def to_jev_response(image_ref, answers, model_name="needle-see-3-graft"):
    return {
        "model": model_name,
        "answers": answers,
        "usage": {"image_inputs": 1},
    }
