
# Project Nucleus v0.7 — Image-Text Alignment

Project Nucleus v0.7 introduces a CLIP-style image-text alignment workflow. It maps image and text inputs into a shared embedding space using symmetric contrastive learning.

## Features

- CNN-based image encoder using Conv2D, ReLU and AvgPool2D.
- Embedding-based text encoder with mean pooling.
- Separate image and text projection heads.
- L2-normalized cosine similarity matrix.
- Symmetric image-to-text and text-to-image contrastive loss.
- Bidirectional Recall@K retrieval metrics.
- End-to-end autograd and SGD training support.

## Architecture

**Image path**

NCHW images → CNN encoder → Global average pooling → Image projection

**Text path**

Token IDs → Embedding → Mean pooling → Text projection

Both projection heads produce embeddings of the same dimension.

## Quick Start

Run the synthetic training demonstration:

```bash
python examples/image_text_alignment_demo.py
```

Customize training:

```bash
python examples/image_text_alignment_demo.py --epochs 200 --learning-rate 0.001
```

The demo uses a small synthetic paired dataset. Reduced training loss demonstrates optimization on that batch, not real-world retrieval accuracy or generalization.

## Components

- `ImageEncoder`: CNN image feature extractor.
- `TextEncoder`: embedding and mean-pooling text encoder.
- `ImageTextAlignmentModel`: combines encoders and projection heads.
- `cosine_similarity_matrix`: computes pairwise cosine similarities.
- `symmetric_contrastive_loss`: computes bidirectional contrastive loss.
- `recall_at_k`: computes bidirectional retrieval metrics.

## Limitations

- No tokenizer or production dataset loader is included.
- Text encoding currently uses embedding and mean pooling, not a Transformer.
- The demo uses synthetic data, not a real image-caption dataset.
- Evaluate retrieval performance on held-out paired examples.

## Tests

Run the full test suite:

```bash
pytest -q
```
