"""Differentiable CLIP-style similarity and symmetric contrastive loss."""

from __future__ import annotations

import numpy as np

from nucleus.core.tensor import Tensor


def _normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """L2-normalize embeddings along their feature dimension."""
    squared = x * x
    norm = (squared.sum(axis=-1, keepdims=True) + eps) ** 0.5
    return x / norm


def cosine_similarity_matrix(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
) -> Tensor:
    """Return pairwise cosine similarities with shape (B_image, B_text)."""
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 (batch, features).")

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("Image and text embedding dimensions must match.")

    image = _normalize(image_embeddings)
    text = _normalize(text_embeddings)

    return image @ text.transpose()


def _logsumexp_rows(logits: Tensor) -> Tensor:
    """Numerically stable row-wise log-sum-exp."""
    row_max = Tensor(
        np.max(logits.data, axis=1, keepdims=True),
        requires_grad=False,
    )
    shifted = logits - row_max
    return shifted.exp().sum(axis=1, keepdims=True).log() + row_max


def _row_cross_entropy(logits: Tensor) -> Tensor:
    """Mean cross-entropy with diagonal positive-pair targets."""
    batch = logits.shape[0]

    if logits.ndim != 2 or logits.shape[1] != batch:
        raise ValueError("Contrastive logits must be square.")

    log_probs = logits - _logsumexp_rows(logits)
    targets = np.eye(batch, dtype=np.float64)

    return -(log_probs * targets).sum(axis=(0, 1)) / batch


def symmetric_contrastive_loss(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Calculate symmetric image-to-text and text-to-image contrastive loss."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 (batch, features).")

    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError("Image and text batch sizes must match.")

    if image_embeddings.shape[0] == 0:
        raise ValueError("Contrastive batch must not be empty.")

    logits = cosine_similarity_matrix(
        image_embeddings,
        text_embeddings,
    ) / temperature

    image_to_text = _row_cross_entropy(logits)
    text_to_image = _row_cross_entropy(logits.transpose())

    return (image_to_text + text_to_image) / 2.0
