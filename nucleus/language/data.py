"""Sliding-window next-character dataset and batching."""
import numpy as np

class CharacterDataset:
    def __init__(self, token_ids, context_length: int):
        tokens = np.asarray(token_ids, dtype=np.int64)
        if tokens.ndim != 1:
            raise ValueError("token_ids must be one-dimensional.")
        if context_length <= 0 or len(tokens) <= context_length:
            raise ValueError("Need positive context_length shorter than token sequence.")
        self.tokens, self.context_length = tokens.copy(), int(context_length)
    def __len__(self):
        return len(self.tokens) - self.context_length
    def __getitem__(self, index):
        if index < 0: index += len(self)
        if index < 0 or index >= len(self): raise IndexError(index)
        end = index + self.context_length
        return self.tokens[index:end].copy(), self.tokens[index+1:end+1].copy()

def batch_iterator(dataset, batch_size: int, shuffle=False, seed=None, drop_last=False):
    if batch_size <= 0: raise ValueError("batch_size must be positive.")
    indices = np.arange(len(dataset))
    if shuffle: np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start+batch_size]
        if drop_last and len(chosen) < batch_size: break
        pairs = [dataset[int(i)] for i in chosen]
        yield np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
