"""Small spectrogram-based audio classification model."""

from __future__ import annotations

from nucleus.core.tensor import Tensor
from nucleus.nn.layers import Linear, Module, ReLU


class AudioEncoder(Module):
    """Encode each spectrogram frame, then mean-pool across time."""

    def __init__(self, n_fft: int = 256, hidden_dim: int = 32, seed: int = 31):
        if n_fft < 2 or hidden_dim < 1:
            raise ValueError("n_fft must be >= 2 and hidden_dim must be positive")
        self.n_fft = n_fft
        self.frame_projection = Linear(n_fft // 2 + 1, hidden_dim, seed=seed)
        self.activation = ReLU()

    def __call__(self, spectrograms: Tensor) -> Tensor:
        if spectrograms.ndim != 3:
            raise ValueError("spectrograms must have shape (batch, frequency_bins, frames)")
        if spectrograms.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(
                f"expected {self.n_fft // 2 + 1} frequency bins, got {spectrograms.shape[1]}"
            )
        if spectrograms.shape[2] == 0:
            raise ValueError("spectrograms must contain at least one frame")
        frames = spectrograms.transpose(0, 2, 1)
        features = self.activation(self.frame_projection(frames))
        return features.mean(axis=1)


class AudioClassifier(Module):
    """Trainable classifier for mono audio spectrograms."""

    def __init__(
        self,
        n_fft: int = 256,
        hidden_dim: int = 32,
        num_classes: int = 2,
        seed: int = 31,
    ):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.encoder = AudioEncoder(n_fft=n_fft, hidden_dim=hidden_dim, seed=seed)
        self.classifier = Linear(hidden_dim, num_classes, seed=seed + 1)

    def __call__(self, spectrograms: Tensor) -> Tensor:
        return self.classifier(self.encoder(spectrograms))
