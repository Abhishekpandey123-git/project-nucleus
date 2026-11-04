"""Multimodal image-text alignment components."""

from .loss import cosine_similarity_matrix, symmetric_contrastive_loss
from .metrics import recall_at_k
from .model import ImageEncoder, ImageTextAlignmentModel, TextEncoder

__all__ = [
    "ImageEncoder",
    "ImageTextAlignmentModel",
    "TextEncoder",
    "cosine_similarity_matrix",
    "recall_at_k",
    "symmetric_contrastive_loss",
]
