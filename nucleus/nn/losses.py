"""Loss functions for supervised learning."""

import numpy as np

from nucleus.core.tensor import Tensor


def mean_squared_error(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return the mean squared error between predictions and targets."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape; got "
            f"{predictions.shape} and {targets.shape}."
        )
    return ((predictions - targets) ** 2).mean()


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Return stable mean cross-entropy for 2-D logits and integer labels."""
    labels = np.asarray(targets, dtype=np.int64)

    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must be a non-empty 2-D tensor.")

    if labels.ndim != 1 or len(labels) != logits.shape[0]:
        raise ValueError("targets must have shape (batch,) or (batch, 1).")

    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("target class outside logits vocabulary.")

    shifted = logits - np.max(logits.data, axis=1, keepdims=True)
    log_partition = shifted.exp().sum(axis=1, keepdims=True).log()
    one_hot = Tensor(np.eye(logits.shape[1])[labels])
    selected_logits = (shifted * one_hot).sum(axis=1, keepdims=True)

    return (log_partition - selected_logits).mean()
