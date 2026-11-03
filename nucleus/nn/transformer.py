"""Educational Transformer components."""
import numpy as np
from nucleus.core.tensor import Tensor
from nucleus.nn.attention import MultiHeadAttention
from nucleus.nn.layers import Linear, Module
from nucleus.nn.parameter import Parameter


class PositionalEncoding(Module):
    def __init__(self, model_dim: int, max_length: int = 512):
        if model_dim <= 0 or max_length <= 0:
            raise ValueError("dimensions must be positive.")
        pos = np.arange(max_length, dtype=float)[:, None]
        dim = np.arange(model_dim, dtype=float)[None, :]
        angle = pos / np.power(10000.0, (2 * np.floor(dim / 2)) / model_dim)
        table = np.empty((max_length, model_dim))
        table[:, 0::2] = np.sin(angle[:, 0::2])
        table[:, 1::2] = np.cos(angle[:, 1::2])
        self.encoding = table[None, :, :]

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.encoding.shape[-1]:
            raise ValueError("expected (batch, sequence, model_dim).")
        if inputs.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence exceeds max_length.")
        return inputs + self.encoding[:, :inputs.shape[1], :]


class LayerNorm(Module):
    def __init__(self, features: int, epsilon: float = 1e-5):
        if features <= 0 or epsilon <= 0:
            raise ValueError("features and epsilon must be positive.")
        self.features, self.epsilon = int(features), float(epsilon)
        self.gamma = Parameter(np.ones((1, 1, features)))
        self.beta = Parameter(np.zeros((1, 1, features)))

    def __call__(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.features:
            raise ValueError("LayerNorm input feature dimension mismatch.")
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return ((x - mean) / (var + self.epsilon) ** 0.5) * self.gamma + self.beta


class FeedForward(Module):
    def __init__(self, model_dim: int, hidden_dim: int, seed=None):
        if model_dim <= 0 or hidden_dim <= 0:
            raise ValueError("dimensions must be positive.")
        seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(2)]
        self.linear1 = Linear(model_dim, hidden_dim, seed=seeds[0])
        self.linear2 = Linear(hidden_dim, model_dim, seed=seeds[1])

    def __call__(self, x: Tensor) -> Tensor:
        return self.linear2(self.linear1(x).relu())


class TransformerBlock(Module):
    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int, seed=None):
        seeds = [int(x.generate_state(1)[0]) for x in np.random.SeedSequence(seed).spawn(2)]
        self.attention_norm = LayerNorm(model_dim)
        self.attention = MultiHeadAttention(model_dim, num_heads, seed=seeds[0])
        self.feedforward_norm = LayerNorm(model_dim)
        self.feedforward = FeedForward(model_dim, feedforward_dim, seed=seeds[1])

    def __call__(self, x: Tensor, causal: bool = True) -> Tensor:
        x = x + self.attention(self.attention_norm(x), causal=causal)
        return x + self.feedforward(self.feedforward_norm(x))
