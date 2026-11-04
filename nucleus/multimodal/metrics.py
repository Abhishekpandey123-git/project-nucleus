"""Non-differentiable retrieval metrics implemented with NumPy."""

from __future__ import annotations

import numpy as np


def recall_at_k(similarity, ks=(1, 5, 10)):
    """Compute image-to-text and text-to-image Recall@K.

    The diagonal represents the ground-truth paired examples.

    Args:
        similarity: Finite square (N, N) similarity matrix.
        ks: Iterable of positive integer retrieval cutoffs.

    Returns:
        Dictionary containing image-to-text and text-to-image Recall@K.
    """
    scores = np.asarray(similarity, dtype=float)

    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("similarity must be a square (N, N) matrix")

    n = scores.shape[0]

    if n == 0:
        raise ValueError("similarity matrix must not be empty")

    if not np.isfinite(scores).all():
        raise ValueError("similarity matrix must contain only finite values")

    try:
        cutoffs = tuple(ks)
    except TypeError as exc:
        raise ValueError("ks must be an iterable of positive integers") from exc

    if not cutoffs:
        raise ValueError("ks must not be empty")

    if any(
        isinstance(k, (bool, np.bool_))
        or not isinstance(k, (int, np.integer))
        or k < 1
        for k in cutoffs
    ):
        raise ValueError("ks must contain positive integers")

    result = {}

    for k in cutoffs:
        cutoff = min(k, n)

        image_to_text = np.argsort(-scores, axis=1)[:, :cutoff]
        text_to_image = np.argsort(-scores.T, axis=1)[:, :cutoff]

        truth = np.arange(n)

        result[f"image_to_text_R@{k}"] = float(
            np.mean(np.any(image_to_text == truth[:, None], axis=1))
        )
        result[f"text_to_image_R@{k}"] = float(
            np.mean(np.any(text_to_image == truth[:, None], axis=1))
        )

    return result