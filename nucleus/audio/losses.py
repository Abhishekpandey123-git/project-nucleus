"""Classification losses for audio models."""

from __future__ import annotations

import numpy as np

from nucleus.core.tensor import Tensor


def cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Numerically stable mean cross-entropy for integer class labels."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (non-empty batch, at least 2 classes)")
    targets = np.asarray(labels)
    if targets.ndim != 1 or len(targets) != logits.shape[0]:
        raise ValueError("labels must be 1-D with one target per logit row")
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("labels must contain integer class indices")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("label index outside class range")

    shifted = logits - np.max(logits.data, axis=1, keepdims=True)
    exp_values = shifted.exp()
#    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[targets]
    # log-softmax is evaluated through Tensor operations to preserve autograd.
    log_probs = shifted - exp_values.sum(axis=1, keepdims=True).log()
    return -(log_probs * one_hot).sum(axis=(0, 1)) / logits.shape[0]


def accuracy(logits: Tensor, labels: np.ndarray) -> float:
    targets = np.asarray(labels)
    if logits.ndim != 2 or targets.ndim != 1 or len(targets) != logits.shape[0]:
        raise ValueError("expected logits (batch, classes) and labels (batch,)")
    return float(np.mean(np.argmax(logits.data, axis=1) == targets))
