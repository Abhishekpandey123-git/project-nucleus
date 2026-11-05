"""Audio preprocessing and a lightweight in-memory dataset."""

from __future__ import annotations

import numpy as np


def normalize_waveforms(waveforms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Peak-normalize each waveform independently without modifying the input."""
    values = np.asarray(waveforms, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("waveforms must have shape (batch, samples) with non-empty samples")
    if not np.isfinite(values).all():
        raise ValueError("waveforms must contain only finite values")
    peak = np.max(np.abs(values), axis=1, keepdims=True)
    return values / np.maximum(peak, eps)


def stft(
    waveforms: np.ndarray,
    n_fft: int = 256,
    hop_length: int = 128,
    window: str = "hann",
) -> np.ndarray:
    """Compute a real-valued magnitude spectrogram.

    Returns an array shaped (batch, frequency_bins, frames). Short signals are
    zero-padded to one complete frame. This is a NumPy preprocessing operation,
    not a differentiable layer.
    """
    values = np.asarray(waveforms, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("waveforms must have shape (batch, samples) with non-empty samples")
    if not np.isfinite(values).all():
        raise ValueError("waveforms must contain only finite values")
    if not isinstance(n_fft, (int, np.integer)) or isinstance(n_fft, bool) or n_fft < 2:
        raise ValueError("n_fft must be an integer >= 2")
    if not isinstance(hop_length, (int, np.integer)) or isinstance(hop_length, bool) or hop_length < 1:
        raise ValueError("hop_length must be a positive integer")
    if window not in {"hann", "rect"}:
        raise ValueError("window must be 'hann' or 'rect'")

    if values.shape[1] < n_fft:
        values = np.pad(values, ((0, 0), (0, n_fft - values.shape[1])))
    frame_count = 1 + (values.shape[1] - n_fft) // hop_length
    frames = np.stack(
        [values[:, i * hop_length:i * hop_length + n_fft] for i in range(frame_count)],
        axis=1,
    )
    win = np.hanning(n_fft) if window == "hann" else np.ones(n_fft)
    spectrum = np.fft.rfft(frames * win[None, None, :], axis=-1)
    return np.abs(spectrum).transpose(0, 2, 1)


class AudioDataset:
    """Small in-memory audio classification dataset.

    Audio is represented as mono waveforms with shape (samples, time). The
    dataset computes spectrograms when batches are requested.
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        labels: np.ndarray,
        n_fft: int = 256,
        hop_length: int = 128,
        normalize: bool = True,
    ):
        values = np.asarray(waveforms, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.int64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("waveforms must have shape (non-empty samples, time)")
        if targets.ndim != 1 or len(targets) != len(values):
            raise ValueError("labels must be a 1-D array with one label per waveform")
        if np.any(targets < 0):
            raise ValueError("labels must be non-negative integers")
        if not np.isfinite(values).all():
            raise ValueError("waveforms must contain only finite values")
        self.waveforms = normalize_waveforms(values) if normalize else values.copy()
        self.labels = targets.copy()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Validate preprocessing settings at construction time.
        stft(self.waveforms[:1], n_fft=n_fft, hop_length=hop_length)

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1

    def batches(self, batch_size: int = 16, shuffle: bool = True, seed: int | None = None):
        """Yield (spectrogram_batch, label_batch) NumPy arrays."""
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        indices = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(self), batch_size):
            selected = indices[start:start + batch_size]
            yield (
                stft(self.waveforms[selected], self.n_fft, self.hop_length),
                self.labels[selected],
            )
