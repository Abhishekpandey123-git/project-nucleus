"""Audio learning components for Project Nucleus v0.8."""

from nucleus.audio.data import AudioDataset, normalize_waveforms, stft
from nucleus.audio.losses import accuracy, cross_entropy
from nucleus.audio.models import AudioClassifier, AudioEncoder

__all__ = [
    "AudioClassifier",
    "AudioDataset",
    "AudioEncoder",
    "accuracy",
    "cross_entropy",
    "normalize_waveforms",
    "stft",
]
