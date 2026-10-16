"""Visual graft for Cactus Needle 3.

:class:`VisualSAN` subclasses the vendored :class:`SimpleAttentionNetwork` and
adds an optional **continuous visual prefix** in front of the byte tokens:

    [BOS] [V visual embeddings] [prompt bytes ...] [target bytes ...]

Design rules
------------
* ``visual_embeds=None`` (or a zero-width prefix) must be numerically identical
  to upstream.  Every override short-circuits to ``super()`` in that case.
* The prefix is *attention-visible*: every text position may attend to the
  visual tokens, and the visual tokens attend to each other causally.
* The prefix is *engram-invisible*: The byte-level n-gram memory is meaningless
  for continuous vectors, so any engram window that intersects the prefix is
  zero-masked, exactly like a window over padding.
* The prefix is *pool-invisible* by default: the probe heads (confidence,
  embedding, router) pool over byte tokens only, so an untrained head never
  reads an out-of-distribution hidden state.  Set ``HEAD_SEES_VISUAL = True``
  to include them.
* ``visual_embeds`` are accepted already scaled by ``sqrt(d_model)`` so the
  projector output lives in the same magnitude space as ``_input_embeddings``.

The caller passes a *text-only* token tensor; prefix placeholders (``PAD_ID``)
are inserted internally for the engram lookups.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from needle.model.architecture import (
    ENGRAM_CONV_TAPS,
    SimpleAttentionNetwork,
    _aq,
    _mask_diag,
    engram_geometry,
    engram_indices,
    ladder_row_keep,
    make_causal_mask,
)


class VisualSAN(SimpleAttentionNetwork):
    """SimpleAttentionNetwork with a continuous visual prefix."""

    HEAD_SEES_VISUAL = False

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _visual_prefix(self, visual_embeds):
        return 0 if visual_embeds is None else int(visual_embeds.shape[1])

    def _visual_tokens(self, tokens, prefix_len):
        """Token tensor with placeholder ids at the visual positions."""
        if not prefix_len:
            return tokens
        pad = jnp.full((tokens.shape[0], prefix_len), self.config.pad_token_id,
                       tokens.dtype)
        return jnp.concatenate([pad, tokens], axis=1)

    # ------------------------------------------------------------------ #
    # engram, prefix aware
    # ------------------------------------------------------------------ #
    def _engram_masks(self, mask, token_len, prefix_len=0):
        """(ngram_ok, tap_ok) for the byte engram memory.

        Entries whose window intersects the visual prefix are zeroed, exactly
        like windows over padding.  With ``prefix_len == 0`` this reproduces the
        vendored computation bit for bit.
        """
        orders, heads, _ = engram_geometry(self.config)
        ngram_ok = jnp.stack(
            [_mask_diag(mask, o - 1) for o in orders for _ in range(heads)], axis=-1)
        tap_ok = jnp.stack(
            [_mask_diag(mask, j * max(orders)) for j in range(ENGRAM_CONV_TAPS)])
        if prefix_len:
            # window [t-o+1, t] lies fully in the text region <=> t >= prefix+o-1
            pos = jnp.arange(token_len)
            ngram_off = jnp.asarray([o - 1 for o in orders for _ in range(heads)])
            ngram_ok = ngram_ok * (
                pos[:, None] >= (prefix_len + ngram_off)[None, :]).astype(ngram_ok.dtype)
            tap_off = jnp.asarray([j * max(orders) for j in range(ENGRAM_CONV_TAPS)])
            tap_ok = tap_ok * (
                pos[None, :] >= (prefix_len + tap_off)[:, None, None]).astype(tap_ok.dtype)
        return ngram_ok, tap_ok

    def _engram_kv(self, tokens, mask, quant, prefix_len=0):
        if not self.engrams:
            return None
        if not prefix_len:
            return super()._engram_kv(tokens, mask, quant)
        orders, heads, _ = engram_geometry(self.config)
        indices = engram_indices(tokens, orders, heads, self.config.engram_slots,
                                 getattr(self.config, "engram_seed_heads", 0))
        ngram_ok, tap_ok = self._engram_masks(mask, tokens.shape[1], prefix_len)
        pairs = [e(indices, ngram_ok, tap_ok, quant=quant) for e in self.engrams]
        return (jnp.stack([k for k, _ in pairs]),
                jnp.stack([v for _, v in pairs]))

    # ------------------------------------------------------------------ #
    # forward paths
    # ------------------------------------------------------------------ #
    def __call__(self, tokens, mask=None, quant=False,
                 exit_depth=None, exit_only=False, subnetwork_only=False,
                 visual_embeds=None):
        prefix = self._visual_prefix(visual_embeds)
        x = self._input_embeddings(tokens)
        if prefix:
            x = jnp.concatenate([visual_embeds.astype(x.dtype), x], axis=1)
            tokens = self._visual_tokens(tokens, prefix)
        seq_len = tokens.shape[1]
        if mask is None:
            mask = make_causal_mask(seq_len)
        rope = self._rope(seq_len)
        engram_kv = self._engram_kv(tokens, mask, quant, prefix_len=prefix)
        stack_out = self.stack(
            x, mask=mask, rope=rope, engram_kv=engram_kv, quant=quant,
            exit_depth=exit_depth, subnetwork_only=subnetwork_only)
        if subnetwork_only:
            x, _ = stack_out
            exit_x = None
        elif exit_depth is None:
            x, _ = stack_out
            exit_x = None
        else:
            x, _, exit_x = stack_out
        head = self.embedding.embedding
        if self.config.out_vocab:
            head = head[: self.config.out_vocab]
        if subnetwork_only:
            return _aq(x, quant).astype(jnp.float32) @ head.T
        if exit_x is not None:
            exit_logits = _aq(exit_x, quant).astype(jnp.float32) @ head.T
            if exit_only:
                return exit_logits
        logits = _aq(x, quant).astype(jnp.float32) @ head.T
        return (logits, exit_logits) if exit_x is not None else logits

    def hidden_cells(self, tokens, quant=False, window=0, sink=None,
                     exit_depth=None, visual_embeds=None):
        cfg = self.config
        prefix = self._visual_prefix(visual_embeds)
        text_valid = tokens != cfg.pad_token_id
        x0 = self._input_embeddings(tokens)
        if prefix:
            x0 = jnp.concatenate([visual_embeds.astype(x0.dtype), x0], axis=1)
            tokens = self._visual_tokens(tokens, prefix)
            valid = jnp.concatenate(
                [jnp.ones_like(text_valid[:, :prefix]), text_valid], axis=1)
        else:
            valid = text_valid
        mask = make_causal_mask(tokens.shape[1]) & valid[:, None, None, :]
        if window:
            pos = jnp.arange(tokens.shape[1])
            recent = ((pos[:, None] - pos[None, :]) < window)[None, None, :, :]
            keep = recent if sink is None else (recent | sink[:, None, None, :])
            mask = mask & keep
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, quant, prefix_len=prefix)
        hidden = self.stack(
            x0, mask=mask, rope=rope, engram_kv=engram_kv,
            quant=quant, collect_hidden=True, exit_depth=exit_depth)[1]
        return jnp.stack([x0, *hidden], axis=2)

    def _head_cells(self, tokens, quant=False, window=0, sink=None,
                    exit_depth=None, visual_embeds=None):
        cells = jax.lax.stop_gradient(
            self.hidden_cells(tokens, quant=quant, window=window, sink=sink,
                              exit_depth=exit_depth, visual_embeds=visual_embeds))
        prefix = self._visual_prefix(visual_embeds)
        keep = (tokens != self.config.pad_token_id).astype(jnp.float32)
        if prefix:
            fill = 1.0 if self.HEAD_SEES_VISUAL else 0.0
            keep = jnp.concatenate(
                [jnp.full((tokens.shape[0], prefix), fill, keep.dtype), keep], axis=1)
        return cells, keep, ladder_row_keep(self.config, exit_depth)

    def forward_embedding(self, tokens, quant=False, window=0, sink=None,
                          exit_depth=None, visual_embeds=None):
        return self.embedding_head(
            *self._head_cells(tokens, quant, window, sink, exit_depth,
                              visual_embeds=visual_embeds), quant=quant)[0]

    def forward_confidence(self, tokens, quant=False, window=0, sink=None,
                           exit_depth=None, visual_embeds=None):
        return self.confidence_head(
            *self._head_cells(tokens, quant, window, sink, exit_depth,
                              visual_embeds=visual_embeds), quant=quant)[..., 0]

    def forward_router(self, tokens, quant=False, window=0, sink=None,
                       exit_depth=None, visual_embeds=None):
        return self.router_head(
            *self._head_cells(tokens, quant, window, sink, exit_depth,
                              visual_embeds=visual_embeds), quant=quant)

    def hidden_states(self, tokens, mask=None, visual_embeds=None):
        prefix = self._visual_prefix(visual_embeds)
        x = self._input_embeddings(tokens)
        if prefix:
            x = jnp.concatenate([visual_embeds.astype(x.dtype), x], axis=1)
            tokens = self._visual_tokens(tokens, prefix)
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, False, prefix_len=prefix)
        _, hidden = self.stack(x, mask=mask, rope=rope, engram_kv=engram_kv,
                               collect_hidden=True)
        return hidden.astype(jnp.float32)
