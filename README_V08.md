# Project Nucleus v0.8 — Audio Learning

This add-on introduces a lightweight audio classification pipeline built on
Project Nucleus' NumPy/autograd foundation.

## Quick start

Copy the `nucleus/audio/`, `examples/audio_classification_demo.py`, and
`tests/test_audio_v08.py` files into the matching locations in your existing
Project Nucleus repository.

Run:

```bash
python examples/audio_classification_demo.py --epochs 25
pytest -q
```

## Features

- Waveform peak normalization
- STFT magnitude spectrogram extraction
- AudioDataset and mini-batch iteration
- Trainable spectrogram AudioEncoder
- AudioClassifier, cross-entropy and accuracy
- Synthetic two-class tone classification demo

See `docs/v0.8-audio-learning.md` for architecture and limitations.

**Compatibility:** designed for the Project Nucleus main branch containing
v0.7 image-text alignment. It expects the existing `nucleus.optim.SGD` API.
