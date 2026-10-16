"""Sanity tests for the synthetic generator ( labels exactly match the image meta)."""

from __future__ import annotations

from PIL import Image

from needlesee.data.gen import build_example, TASK_NAMES
from needlesee.questions import TOOLS, render_call


def test_each_task_renders_an_image():
    for name in TASK_NAMES:
        example = build_example(seed=42, index=0, task=name)
        assert isinstance(example.image, Image.Image)
        assert example.image.size == (256, 256)
        assert example.image.mode == "RGB"


def test_truth_index_maps_to_candidates_consistently():
    for idx, name in enumerate(TASK_NAMES):
        example = build_example(seed=99, index=idx, task=name)
        q = example.question
        candidates = q.candidate_texts()
        truth_text = render_call(q.answers(example.truth_index)[0])
        assert truth_text in candidates
        again = build_example(seed=99, index=idx, task=name)
        assert again.truth_index == example.truth_index
        assert again.question.query() == example.question.query()


def test_tools_are_reused():
    assert isinstance(TOOLS, list)
    assert "choose" in {t["name"] for t in TOOLS}


def test_image_question_jsonl_round_trip_shape():
    from needlesee.paths import ensure_needle; ensure_needle()
    from needle.model.finetune import render_example
    example = build_example(seed=7, index=0, task="color_of_shape")
    row = {"query": example.question.query(), "tools": TOOLS,
           "answers": example.question.answers(example.truth_index)}
    prompt, target = render_example(row)
    assert "<tools>" in prompt
    assert "<tool_call>" in target
