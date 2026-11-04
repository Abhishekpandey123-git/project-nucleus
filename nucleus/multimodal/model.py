"""Image and text encoders for multimodal alignment."""

from nucleus.core.tensor import Tensor
from nucleus.nn.embedding import Embedding
from nucleus.nn.layers import Module, ReLU
from nucleus.vision.layers import AvgPool2D, Conv2D


class ImageEncoder(Module):
    """CNN encoder with global average pooling."""

    def __init__(self, in_channels=1, feature_dim=16, seed=7):
        self.conv1 = Conv2D(in_channels, 8, 3, seed=seed)
        self.relu1 = ReLU()
        self.pool = AvgPool2D(2)
        self.conv2 = Conv2D(8, feature_dim, 3, seed=seed + 1)
        self.relu2 = ReLU()

    def __call__(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError("Expected NCHW image input.")
        if images.shape[2] < 8 or images.shape[3] < 8:
            raise ValueError("Image dimensions must be at least 8x8.")

        x = self.conv1(images)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x.mean(axis=(2, 3))


class TextEncoder(Module):
    """Embedding-based text encoder with mean pooling."""

    def __init__(self, vocab_size, embedding_dim=32, seed=17):
        self.embedding = Embedding(vocab_size, embedding_dim, seed=seed)

    def __call__(self, token_ids) -> Tensor:
        steps = self.embedding(token_ids)
        if not steps:
            raise ValueError("Text sequence cannot be empty.")

        pooled = steps[0]
        for step in steps[1:]:
            pooled = pooled + step

        return pooled / len(steps)


class ImageTextAlignmentModel(Module):
    """Combine image/text encoders and projection heads."""

    def __init__(
        self,
        image_encoder,
        text_encoder,
        image_projection,
        text_projection,
    ):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.text_projection = text_projection

    def encode_image(self, images):
        return self.image_projection(self.image_encoder(images))

    def encode_text(self, token_ids):
        return self.text_projection(self.text_encoder(token_ids))

    def __call__(self, images, token_ids):
        return self.encode_image(images), self.encode_text(token_ids)
