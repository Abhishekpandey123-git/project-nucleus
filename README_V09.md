# Project Nucleus v0.9 — Multimodal Neural Playground

v0.9 adds a dependency-light browser interface on top of the existing NumPy neural learning engine.

## Run

```bash
python -m pip install -e ".[dev]"
python examples/web_app_v09.py
```

Open http://127.0.0.1:8000 in a browser.

## Browser features

- **Text:** train a tiny character RNN on supplied text and generate continuations.
- **Image:** upload an image; the browser converts it to 8×8 grayscale and the Nucleus vision model classifies simple learned horizontal/vertical patterns.
- **Audio:** upload audio; the browser decodes PCM and the Nucleus audio model classifies simple 440 Hz / 880 Hz tone patterns using STFT preprocessing.
- **About:** shows the v0.9 modality status.

## Architecture

```text
Browser
   |
   v
Python stdlib HTTP server
   |
   +--> Text -> Character RNN -> Nucleus autograd
   +--> Image -> SmallCNN -> Nucleus autograd
   +--> Audio -> STFT -> AudioClassifier -> Nucleus autograd
```

No Flask, FastAPI, PyTorch, TensorFlow, or frontend framework is required for the demo.

## Scope

v0.9 is an educational user interface and demonstration layer. It does not claim production-grade speech recognition, image understanding, or general-purpose multimodal intelligence.
