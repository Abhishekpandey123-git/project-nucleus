"""Train a small CNN and text encoder using contrastive alignment."""

import argparse

import numpy as np

from nucleus.core.tensor import Tensor
from nucleus.multimodal import (
    ImageEncoder,
    ImageTextAlignmentModel,
    TextEncoder,
    symmetric_contrastive_loss,
)
from nucleus.nn.layers import Linear
from nucleus.optim.sgd import SGD


def build_model(vocab_size=30, projection_dim=8, seed=42):
    """Build the image-text alignment model."""
    return ImageTextAlignmentModel(
        image_encoder=ImageEncoder(1, 16, seed=seed),
        text_encoder=TextEncoder(vocab_size, 32, seed=seed + 1),
        image_projection=Linear(16, projection_dim, seed=seed + 2),
        text_projection=Linear(32, projection_dim, seed=seed + 3),
    )


def train_demo(epochs=100, learning_rate=0.001, seed=42):
    """Train on a deterministic synthetic paired dataset."""
    rng = np.random.default_rng(seed)

    model = build_model(seed=seed)
    optimizer = SGD(model.parameters(), learning_rate=learning_rate)

    images = Tensor(rng.normal(size=(4, 1, 12, 12)))
    token_ids = rng.integers(0, 30, size=(4, 6))

    def calculate_loss():
        image_embeddings, text_embeddings = model(images, token_ids)
        return symmetric_contrastive_loss(
            image_embeddings,
            text_embeddings,
            temperature=0.07,
        )

    initial_loss = float(calculate_loss().data)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = calculate_loss()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: loss={float(loss.data):.6f}")

    final_loss = float(calculate_loss().data)

    print("\nTraining summary")
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Loss reduced: {final_loss < initial_loss}")

    return model, initial_loss, final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Project Nucleus v0.7 image-text alignment demo."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    train_demo(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
