"""Trainable projector from frozen CLIP patch features into Needle embedding space.

The projector output is consumed by :class:`needlesee.model3.VisualSAN` as an
already-scaled prefix (the network scales byte embeddings by ``sqrt(d_model)``,
so the projector does too).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn


class VisualProjector(nn.Module):
    """(B, num_tokens, feature_dim) -> (B, num_tokens, d_model) scaled prefix.

    A two-layer MLP with LayerNorm on the raw features and a learned position
    embedding.  Output magnitude is anchored to the byte-embedding scale by the
    final ``sqrt(d_model)`` factor.
    """

    d_model: int
    num_tokens: int = 49
    feature_dim: int = 768
    hidden: int = 1024
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, features):
        x = features.astype(self.dtype)
        x = nn.LayerNorm(dtype=self.dtype, name="in_norm")(x)
        x = nn.Dense(self.hidden, dtype=self.dtype, name="up")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, dtype=self.dtype, name="down")(x)
        x = nn.LayerNorm(dtype=self.dtype, name="out_norm")(x)
        pos = self.param("pos", jinit.normal(stddev=0.02), (self.num_tokens, self.d_model))
        x = (x + pos) * jnp.asarray(math.sqrt(self.d_model), self.dtype)
        return x.astype(self.dtype)


def init_projector(feature_dim, num_tokens, d_model, hidden, key):
    """Convenience wrapper returning ``(params, module)``."""
    module = VisualProjector(d_model=d_model, num_tokens=num_tokens,
                             feature_dim=feature_dim, hidden=hidden)
    dummy = jnp.zeros((1, num_tokens, feature_dim), jnp.float32)
    params = module.init(key, dummy)
    return params, module
