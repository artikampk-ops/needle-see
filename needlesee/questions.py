"""Typed questions and their exact tool-call renderings.

This module is the single source of truth for how a question becomes (a) the
prompt text the model reads and (b) the candidate completions it must choose
between.  The synthetic generator, the trainer and the runtime all go through
it, so training and inference can never drift apart.

Primitive mapping (Jev-shaped):

* ``ChoiceQuestion`` -> ``choose`` tool  (value + distribution)
* ``JudgeQuestion``  -> ``judge``  tool  (noul-style probability of yes)
* ``RateQuestion``   -> ``rate``   tool  (score level + distribution)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

TOOLS = [
    {
        "name": "choose",
        "description": "Pick exactly one of the declared options.",
        "parameters": {
            "type": "object",
            "properties": {
                "option": {"type": "string", "description": "One declared option, exactly as written."}
            },
            "required": ["option"],
        },
    },
    {
        "name": "judge",
        "description": "Answer a yes/no question about the image.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "boolean"}},
            "required": ["answer"],
        },
    },
    {
        "name": "rate",
        "description": "Give the 0-based index of the single level that best matches the image.",
        "parameters": {
            "type": "object",
            "properties": {"level": {"type": "integer"}},
            "required": ["level"],
        },
    },
]

SYSTEM = "You answer questions about the attached image using the provided tools."


def render_call(call: dict) -> str:
    """The exact completion text for one tool call (matches render_example)."""
    return json.dumps([call], separators=(",", ":"), ensure_ascii=False)


class Question:
    task: str = "?"

    def query(self) -> str:
        raise NotImplementedError

    def candidates(self) -> list[dict]:
        raise NotImplementedError

    def answers(self, truth_index: int) -> list[dict]:
        return [self.candidates()[truth_index]]

    # -- convenience ----------------------------------------------------
    def candidate_texts(self) -> list[str]:
        return [render_call(c) for c in self.candidates()]

    def example(self, truth_index: int) -> dict:
        return {"query": self.query(), "tools": TOOLS, "answers": self.answers(truth_index)}


@dataclass(frozen=True)
class ChoiceQuestion(Question):
    """Pick one option; the result carries a value and a distribution."""

    question: str
    options: tuple[str, ...]
    task: str = "choice"

    def query(self) -> str:
        options = " | ".join(self.options)
        return f"Question about the image: {self.question}\nOptions: {options}"

    def candidates(self) -> list[dict]:
        return [{"name": "choose", "arguments": {"option": o}} for o in self.options]


@dataclass(frozen=True)
class JudgeQuestion(Question):
    """Yes/no; the runtime maps P(yes) to a noul probability."""

    question: str
    task: str = "judge"

    def query(self) -> str:
        return f"Question about the image: {self.question} Answer yes or no."

    def candidates(self) -> list[dict]:
        return [{"name": "judge", "arguments": {"answer": True}},
                {"name": "judge", "arguments": {"answer": False}}]


@dataclass(frozen=True)
class RateQuestion(Question):
    """Ordered levels; the runtime maps the argmax to a score + distribution."""

    question: str
    levels: tuple[str, ...]
    task: str = "rate"

    def query(self) -> str:
        levels = " | ".join(f"{i} = {label}" for i, label in enumerate(self.levels))
        return f"Question about the image: {self.question}\nLevels: {levels}"

    def candidates(self) -> list[dict]:
        return [{"name": "rate", "arguments": {"level": i}}
                for i in range(len(self.levels))]
