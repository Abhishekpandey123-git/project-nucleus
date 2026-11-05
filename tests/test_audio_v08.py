import numpy as np
import pytest

from nucleus.audio import (
    AudioClassifier,
    AudioDataset,
    accuracy,
    cross_entropy,
    normalize_waveforms,
    stft,
)
from nucleus.core.tensor import Tensor


def test_normalize_waveforms_peak_per_example():
    values = np.array([[0.0, 2.0, -1.0], [0.0, 0.0, 0.0]])
    result = normalize_waveforms(values)
    np.testing.assert_allclose(np.max(np.abs(result), axis=1), [1.0, 0.0])
    np.testing.assert_array_equal(values, [[0.0, 2.0, -1.0], [0.0, 0.0, 0.0]])


def test_stft_shape_and_finite_values():
    waves = np.ones((2, 1024))
    result = stft(waves, n_fft=256, hop_length=128)
    assert result.shape == (2, 129, 7)
    assert np.isfinite(result).all()


def test_stft_pads_short_waveform():
    assert stft(np.ones(32), n_fft=128).shape == (1, 65, 1)


def test_dataset_batches_cover_all_samples():
    dataset = AudioDataset(np.ones((5, 512)), np.array([0, 1, 0, 1, 0]), n_fft=128)
    batches = list(dataset.batches(batch_size=2, shuffle=False))
    assert [len(y) for _, y in batches] == [2, 2, 1]


def test_audio_classifier_forward_backward():
    rng = np.random.default_rng(4)
    specs = Tensor(rng.normal(size=(4, 33, 5)))
    model = AudioClassifier(n_fft=64, hidden_dim=12, num_classes=3)
    logits = model(specs)
    assert logits.shape == (4, 3)
    loss = cross_entropy(logits, np.array([0, 1, 2, 1]))
    loss.backward()
    assert np.isfinite(loss.data).all()
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in model.parameters())


def test_cross_entropy_and_accuracy():
    logits = Tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    labels = np.array([0, 1])
    loss = cross_entropy(logits, labels)
    loss.backward()
    assert float(loss.data) < 0.1
    assert accuracy(logits, labels) == 1.0


@pytest.mark.parametrize("kwargs", [
    {"n_fft": 1},
    {"hop_length": 0},
    {"window": "bad"},
])
def test_stft_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        stft(np.ones(128), **kwargs)
