"""Explicitly unrolled tanh recurrent character language model."""
import numpy as np
from nucleus.core.tensor import Tensor
from nucleus.nn.layers import Module
from nucleus.nn.parameter import Parameter
from nucleus.nn.embedding import Embedding

class CharacterLanguageModel(Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=32, seed=None):
        if min(vocab_size, embedding_dim, hidden_size) <= 0:
            raise ValueError("model dimensions must be positive.")
        rng = np.random.default_rng(seed)
        self.vocab_size, self.hidden_size = int(vocab_size), int(hidden_size)
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)
        self.weight_ih = Parameter(rng.normal(0, 1/np.sqrt(embedding_dim), (embedding_dim, hidden_size)))
        self.weight_hh = Parameter(rng.normal(0, 1/np.sqrt(hidden_size), (hidden_size, hidden_size)))
        self.bias_h = Parameter(np.zeros((1, hidden_size)))
        self.weight_out = Parameter(rng.normal(0, 1/np.sqrt(hidden_size), (hidden_size, vocab_size)))
        self.bias_out = Parameter(np.zeros((1, vocab_size)))
    def __call__(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("token_ids must be a non-empty 2-D batch.")
        hidden = Tensor(np.zeros((ids.shape[0], self.hidden_size)))
        outputs = []
        for embedded in self.embedding(ids):
            hidden = (embedded @ self.weight_ih + hidden @ self.weight_hh + self.bias_h).tanh()
            outputs.append(hidden @ self.weight_out + self.bias_out)
        return outputs
