"""Train a tiny audio classifier on synthetic tone/noise examples."""

from __future__ import annotations

import argparse

import numpy as np

from nucleus.audio import AudioClassifier, AudioDataset, accuracy, cross_entropy
from nucleus.core.tensor import Tensor, no_grad
from nucleus.optim import SGD


def make_tone_dataset(samples: int = 96, length: int = 2048, seed: int = 7):
    rng = np.random.default_rng(seed)
    time = np.arange(length) / length
    waveforms = []
    labels = []
    for index in range(samples):
        label = index % 2
        frequency = 8.0 if label == 0 else 24.0
        phase = rng.uniform(0, 2 * np.pi)
        signal = np.sin(2 * np.pi * frequency * time + phase)
        signal += 0.12 * rng.normal(size=length)
        waveforms.append(signal)
        labels.append(label)
    return np.asarray(waveforms), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    waves, labels = make_tone_dataset()
    dataset = AudioDataset(waves, labels, n_fft=256, hop_length=128)
    model = AudioClassifier(n_fft=256, hidden_dim=32, num_classes=2, seed=5)
    optimizer = SGD(model.parameters(), learning_rate=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        losses = []
        for specs, targets in dataset.batches(batch_size=16, shuffle=True, seed=epoch):
            logits = model(Tensor(specs))
            loss = cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))

        with no_grad():
            eval_logits = model(Tensor(stft_for_eval(waves, dataset)))
            score = accuracy(eval_logits, labels)
        print(
            f"epoch={epoch:02d} loss={np.mean(losses):.4f} "
            f"train_accuracy={score:.3f}"
        )


def stft_for_eval(waves, dataset):
    from nucleus.audio import stft
    return stft(waves, dataset.n_fft, dataset.hop_length)


if __name__ == "__main__":
    main()
