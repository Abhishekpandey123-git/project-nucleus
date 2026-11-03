"""Differentiable attention primitives."""
import numpy as np
from nucleus.core.tensor import Tensor
from nucleus.nn.layers import Linear, Module


def softmax(inputs: Tensor, axis: int = -1) -> Tensor:
    if inputs.ndim == 0:
        raise ValueError("softmax requires at least one dimension.")
    axis %= inputs.ndim
    shifted = inputs - np.max(inputs.data, axis=axis, keepdims=True)
    ex = shifted.exp()
    return ex / ex.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                 causal: bool = False):
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("query, key, and value must be 4-D.")
    if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
        raise ValueError("batch and head dimensions must match.")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("incompatible query/key/value dimensions.")
    scores = (query @ key.transpose(0, 1, 3, 2)) / np.sqrt(query.shape[-1])
    if causal:
        q_len = query.shape[-2]
        k_len = key.shape[-2]
        if q_len != k_len:
            raise ValueError("causal attention requires equal sequence lengths.")
        mask = np.triu(np.ones((q_len, k_len)), k=1) * -1e9
        scores = scores + Tensor(mask.reshape(1, 1, q_len, k_len))
    weights = softmax(scores, axis=-1)
    return weights @ value, weights


class MultiHeadAttention(Module):
    def __init__(self, model_dim: int, num_heads: int, seed=None):
        if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
            raise ValueError("model_dim and num_heads must be positive; model_dim must divide evenly.")
        self.model_dim, self.num_heads = int(model_dim), int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(4)]
        self.query = Linear(model_dim, model_dim, seed=seeds[0])
        self.key = Linear(model_dim, model_dim, seed=seeds[1])
        self.value = Linear(model_dim, model_dim, seed=seeds[2])
        self.output = Linear(model_dim, model_dim, seed=seeds[3])

    def __call__(self, inputs: Tensor, causal: bool = False) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.model_dim:
            raise ValueError(f"expected (batch, sequence, {self.model_dim}), got {inputs.shape}.")
        batch, seq, _ = inputs.shape

        def split(x):
            return x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split(self.query(inputs)), split(self.key(inputs)), split(self.value(inputs))
        context, _ = scaled_dot_product_attention(q, k, v, causal=causal)
        merged = context.transpose(0, 2, 1, 3).reshape(batch, seq, self.model_dim)
        return self.output(merged)
