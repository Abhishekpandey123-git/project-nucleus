"""Trainable embedding lookup using the engine's 2-D matrix multiplication."""
import numpy as np
from nucleus.core.tensor import Tensor
from nucleus.nn.layers import Module
from nucleus.nn.parameter import Parameter

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, seed=None):
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive.")
        rng = np.random.default_rng(seed)
        self.weight = Parameter(rng.normal(0, 1/np.sqrt(embedding_dim),
                                           (num_embeddings, embedding_dim)))
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
    def __call__(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2: raise ValueError("token_ids must have shape (batch, sequence_length).")
        if np.any(ids < 0) or np.any(ids >= self.num_embeddings):
            raise ValueError("token ID outside vocabulary.")
        return [Tensor(np.eye(self.num_embeddings)[ids[:, t]]) @ self.weight
                for t in range(ids.shape[1])]
