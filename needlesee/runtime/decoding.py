"""Decoding and candidate scoring for the grafted model.

Two read-outs, both through Needle's own weights:

* :func:`greedy_generate` — free-form greedy decoding of the tool-call text from
  a fixed-size buffer (single JIT compile, CPU-friendly).
* :func:`score_completions` — teacher-forced log-probability of each candidate
  completion, batched and length-bucketed.  Softmax over candidates gives a
  calibrated-shape distribution (choice / noul / score), so the runtime reports
  a distribution and confidence without any extra head.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from needlesee.paths import ensure_needle

ensure_needle()

from needle.model.tokenizer import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    IM_END_ID,
    PAD_ID,
)

DECODE_TOTAL = 512
SCORE_BATCH = 8
SCORE_LEN = 512


def encode_example(tokenizer, example, max_len: int):
    """Mirror of ``needle.model.finetune._encode`` for one example."""
    from needle.model.finetune import _encode
    ids, mask = _encode(tokenizer, example, max_len)
    return np.asarray(ids, np.int32), np.asarray(mask, np.float32)


def visual_embeds(projector, projector_params, features):
    """Run the projector on a (N, 50, feature) or (50, feature) batch -> (N, 50, d_model)."""
    features = np.asarray(features, np.float32)
    if features.ndim == 2:
        features = features[None]
    return projector.apply({"params": projector_params},
                           jnp.asarray(features))


def score_completions(model, params, projector, projector_params, features,
                      tokenizer, prompt_ids, candidate_texts,
                      batch: int = SCORE_BATCH, total_len: int = SCORE_LEN) -> np.ndarray:
    """Sum of log P(completion | prompt, image) for each candidate text.

    All candidates share the prompt; each row is ``[BOS] + prompt + cand + [EOS]``
    and only the candidate tokens (and EOS) contribute to the score, matching the
    training loss mask.
    """
    prompt_ids = list(prompt_ids)
    rows, masks = [], []
    for text in candidate_texts:
        cand = tokenizer.encode(text)
        ids = [BOS_ID] + prompt_ids + cand + [EOS_ID]
        mask = [0.0] * (1 + len(prompt_ids)) + [1.0] * (len(cand) + 1)
        if len(ids) > total_len:
            raise ValueError(f"candidate sequence too long ({len(ids)} > {total_len})")
        pad = total_len - len(ids)
        rows.append(ids + [PAD_ID] * pad)
        masks.append(mask + [0.0] * pad)

    v = visual_embeds(projector, projector_params, features)
    prefix = int(v.shape[1])

    scores = []
    for start in range(0, len(rows), batch):
        chunk_ids = np.asarray(rows[start:start + batch], np.int32)
        chunk_mask = np.asarray(masks[start:start + batch], np.float32)
        chunk_v = jnp.broadcast_to(v, (chunk_ids.shape[0], v.shape[1], v.shape[2]))
        logits = model.apply({"params": params}, jnp.asarray(chunk_ids),
                             visual_embeds=chunk_v)
        logits = logits[:, prefix - 1:prefix - 1 + chunk_ids.shape[1]]
        log_probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)
        targets = jnp.asarray(chunk_ids[:, 1:])
        token_scores = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        weights = jnp.asarray(chunk_mask[:, 1:])
        scores.append(np.asarray((token_scores * weights).sum(axis=1)))
    return np.concatenate(scores)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, np.float64)
    scores = scores - scores.max()
    exp = np.exp(scores)
    return exp / exp.sum()


def greedy_generate(model, params, projector, projector_params, features,
                    tokenizer, prompt_ids, max_new_tokens: int = 96,
                    total_len: int = DECODE_TOTAL, stop_ids=(EOS_ID, IM_END_ID)) -> str:
    """Greedy decode the completion text for one prompt + image."""
    prompt_ids = list(prompt_ids)
    if len(prompt_ids) + max_new_tokens > total_len:
        raise ValueError(f"prompt {len(prompt_ids)} + max_new {max_new_tokens} "
                         f"exceeds buffer {total_len}")
    v = visual_embeds(projector, projector_params, features)
    buffer = np.full((1, total_len), PAD_ID, np.int32)
    buffer[0, :len(prompt_ids)] = prompt_ids
    start = np.int32(len(prompt_ids))
    stops = jnp.asarray(list(stop_ids), jnp.int32)

    @jax.jit
    def run(buf, position):
        def body(carry):
            buf, pos = carry
            logits = model.apply({"params": params}, buf, visual_embeds=v)
            next_id = jnp.argmax(logits[0, pos - 1]).astype(jnp.int32)
            buf = buf.at[0, pos].set(next_id)
            return buf, pos + 1

        def cond(carry):
            buf, pos = carry
            last = buf[0, pos - 1]
            stop = jnp.isin(last, stops)
            return jnp.logical_and(pos < buf.shape[1], jnp.logical_not(stop))

        buf, pos = jax.lax.while_loop(cond, body, (buf, position))
        return buf, pos

    out, end = run(jnp.asarray(buffer), start)
    out = np.asarray(out)[0]
    end = int(np.asarray(end))
    generated = out[len(prompt_ids):end]
    if len(generated) and generated[-1] in stop_ids:
        generated = generated[:-1] if generated[-1] == IM_END_ID else generated
    return tokenizer.decode([int(t) for t in generated if t not in (EOS_ID,)])


def parse_calls(text: str):
    """Tool calls in a generated completion, or None if unparseable."""
    from needle.model.finetune import _calls_of
    from needle.model.tokenizer import TOOL_CALL_END, TOOL_CALL_START
    body = text
    if TOOL_CALL_START not in body:
        return []
    return _calls_of(TOOL_CALL_START + body.split(TOOL_CALL_START, 1)[1].split(TOOL_CALL_END)[0]
                     + TOOL_CALL_END)


def render_call(call: dict) -> str:
    return json.dumps([call], separators=(",", ":"), ensure_ascii=False)
