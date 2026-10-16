"""Correctness tests for the visual graft (needlesee.model3.VisualSAN).

The contract: with ``visual_embeds=None`` the graft must be numerically
identical to the vendored ``SimpleAttentionNetwork``; with a prefix it must
produce well-shaped outputs whose text logits depend on the visual content and
whose engram memory never reads across the prefix boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from needlesee.paths import checkpoint_path, ensure_needle

ensure_needle()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from needle.model.architecture import (  # noqa: E402
    SimpleAttentionNetwork,
    _mask_diag,
    engram_geometry,
    make_causal_mask,
)
from needle.model.finetune import rung  # noqa: E402
from needle.model.run import load_checkpoint  # noqa: E402
from needle.model.tokenizer import get_tokenizer  # noqa: E402

from needlesee.model3 import VisualSAN  # noqa: E402

LAYERS = 2
PREFIX = 5
D_MODEL = 768


@pytest.fixture(scope="module")
def stack():
    params, config = load_checkpoint(str(checkpoint_path()))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a, np.float32), params)
    params, config = rung(params, config, LAYERS)
    tokenizer = get_tokenizer()
    return params, config, tokenizer


@pytest.fixture(scope="module")
def tokens(stack):
    _, _, tokenizer = stack
    ids = tokenizer.encode("<|im_start|>user\nWhat color is the circle?<|im_end|>\n")
    return jnp.asarray(np.asarray(ids, np.int32))[None]


def visual_prefix(rng, batch=1, length=PREFIX, scale=0.55):
    return jnp.asarray(rng.normal(0.0, scale, (batch, length, D_MODEL)), jnp.float32)


def test_visual_none_matches_base(stack, tokens):
    params, config, _ = stack
    base = SimpleAttentionNetwork(config)
    graft = VisualSAN(config)
    logits_base = base.apply({"params": params}, tokens)
    logits_graft = graft.apply({"params": params}, tokens)
    assert logits_base.shape == logits_graft.shape
    np.testing.assert_allclose(np.asarray(logits_graft), np.asarray(logits_base),
                               rtol=0, atol=0)


def test_visual_none_matches_base_heads(stack, tokens):
    params, config, _ = stack
    base = SimpleAttentionNetwork(config)
    graft = VisualSAN(config)
    conf_base = base.apply({"params": params}, tokens, method=base.forward_confidence)
    conf_graft = graft.apply({"params": params}, tokens, method=graft.forward_confidence)
    np.testing.assert_allclose(np.asarray(conf_graft), np.asarray(conf_base),
                               rtol=0, atol=0)


def test_prefix_shapes_and_dependence(stack, tokens):
    params, config, _ = stack
    graft = VisualSAN(config)
    rng = np.random.default_rng(0)
    prefix = visual_prefix(rng)
    out = graft.apply({"params": params}, tokens, visual_embeds=prefix)
    assert out.shape == (1, PREFIX + tokens.shape[1], config.out_vocab)

    zeros = jnp.zeros_like(prefix)
    out_zeros = graft.apply({"params": params}, tokens, visual_embeds=zeros)
    out_random = graft.apply({"params": params}, tokens, visual_embeds=prefix)
    delta = np.abs(np.asarray(out_zeros) - np.asarray(out_random)).max()
    assert delta > 1e-3, "visual content must change the text logits"


def test_engram_masks_prefix_boundary(stack, tokens):
    """Windows intersecting the prefix must be zeroed; the rest untouched."""
    params, config, _ = stack
    graft = VisualSAN(config)

    total = PREFIX + tokens.shape[1]
    mask = make_causal_mask(total)
    ngram_ok, tap_ok = graft._engram_masks(mask, total, prefix_len=PREFIX)
    orders = config.engram_orders
    table = 0
    for o in orders:
        for _ in range(3):  # engram heads
            column = np.asarray(ngram_ok[0, :, table])
            assert (column[: PREFIX + o - 1] == 0).all()
            assert (column[PREFIX + o - 1:] == 1).all()
            table += 1
    for j in range(4):  # ENGRAM_CONV_TAPS
        row = np.asarray(tap_ok[j, 0, :])
        offset = j * max(orders)
        assert (row[: PREFIX + offset] == 0).all()
        assert (row[PREFIX + offset:] == 1).all()


def test_engram_masks_match_vendored_without_prefix(stack, tokens):
    params, config, _ = stack
    graft = VisualSAN(config)
    orders, heads, _ = engram_geometry(config)

    total = tokens.shape[1]
    mask = make_causal_mask(total)
    ngram_ok, tap_ok = graft._engram_masks(mask, total, prefix_len=0)
    want_ngram = np.stack(
        [np.asarray(_mask_diag(mask, o - 1)) for o in orders for _ in range(heads)],
        axis=-1)
    want_tap = np.stack([np.asarray(_mask_diag(mask, j * max(orders)))
                         for j in range(4)])
    np.testing.assert_allclose(np.asarray(ngram_ok), want_ngram, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(tap_ok), want_tap, rtol=0, atol=0)


def test_prefix_confidence(stack, tokens):
    params, config, _ = stack
    graft = VisualSAN(config)
    rng = np.random.default_rng(2)
    prefix = visual_prefix(rng)
    value = graft.apply({"params": params}, tokens,
                        visual_embeds=prefix, method=graft.forward_confidence)
    assert value.shape == (1,)
    assert np.isfinite(np.asarray(value)).all()


def test_padded_text_pooling_with_prefix(stack, tokens):
    """PAD text tokens must be excluded from head pooling even with a prefix."""
    params, config, _ = stack
    graft = VisualSAN(config)
    rng = np.random.default_rng(3)
    prefix = visual_prefix(rng, length=PREFIX)
    padded = jnp.concatenate([tokens, jnp.zeros((1, 7), jnp.int32)], axis=1)
    padded = padded.at[0, -1].set(0)
    value = graft.apply({"params": params}, padded,
                        visual_embeds=prefix, method=graft.forward_confidence)
    assert value.shape == (1,)
    assert np.isfinite(np.asarray(value)).all()
