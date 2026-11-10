"""Dependency-light HTTP server exposing Nucleus demos to a browser."""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from nucleus.audio.data import stft
from nucleus.audio.losses import accuracy, cross_entropy
from nucleus.audio.models import AudioClassifier
from nucleus.core.tensor import Tensor
from nucleus.language.generation import generate
from nucleus.language.model import CharacterLanguageModel
from nucleus.language.tokenizer import CharTokenizer
from nucleus.nn.layers import Linear, Module
from nucleus.optim import SGD
from nucleus.vision.data import normalize_images
from nucleus.vision.losses import cross_entropy as vision_cross_entropy
from nucleus.vision.models import SmallCNN

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"


class _TextBundle:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.corpus = None

    def train(self, corpus: str, epochs: int = 80, seed: int = 7) -> None:
        if len(corpus) < 8:
            raise ValueError("corpus must contain at least 8 characters")
        tokenizer = CharTokenizer(corpus)
        ids = np.asarray(tokenizer.encode(corpus), dtype=np.int64)
        if len(ids) < 4:
            raise ValueError("corpus does not contain enough tokens")
        x = ids[:-1][None, :]
        y = ids[1:][None, :]
        model = CharacterLanguageModel(
            tokenizer.vocab_size,
            embedding_dim=8,
            hidden_size=16,
            seed=seed,
        )
        optimizer = SGD(model.parameters(), learning_rate=0.08)
        for _ in range(max(1, min(int(epochs), 150))):
            outputs = model(x)
            loss = sum(
                _token_cross_entropy(logits, np.asarray([y[0, step]]))
                for step, logits in enumerate(outputs)
            ) / len(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("train the text model first")
        return generate(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=11,
        )


def _token_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    shifted = logits - np.max(logits.data, axis=1, keepdims=True)
    log_probs = shifted - shifted.exp().sum(axis=1, keepdims=True).log()
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[labels]
    return -(log_probs * one_hot).sum() / logits.shape[0]


class _TinyVisionBundle:
    def __init__(self) -> None:
        self.model = SmallCNN(in_channels=1, num_classes=2, seed=19)
        self._train()

    def _train(self) -> None:
        rng = np.random.default_rng(19)
        images = np.zeros((40, 1, 8, 8), dtype=np.float64)
        labels = np.repeat([0, 1], 20)
        for i in range(20):
            images[i, 0, 3:5, :] = 1.0
            images[20 + i, 0, :, 3:5] = 1.0
        images += rng.normal(0.0, 0.04, images.shape)
        images = np.clip(images, 0.0, 1.0)
        x = Tensor(images)
        optimizer = SGD(self.model.parameters(), learning_rate=0.05)
        for _ in range(90):
            loss = vision_cross_entropy(self.model(x), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, image: np.ndarray) -> tuple[int, float]:
        x = normalize_images(np.asarray(image)[None, :, :], scale=False)
        if x.shape != (1, 1, 8, 8):
            raise ValueError("image must be an 8x8 grayscale array")
        logits = self.model(Tensor(x)).data[0]
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(np.argmax(probs)), float(np.max(probs))


class _TinyAudioBundle:
    def __init__(self) -> None:
        self.model = AudioClassifier(n_fft=64, hidden_dim=16, num_classes=2, seed=23)
        self._train()

    def _train(self) -> None:
        rng = np.random.default_rng(23)
        sample_rate = 8000
        t = np.arange(512) / sample_rate
        waves = []
        labels = []
        for label, freq in enumerate((440.0, 880.0)):
            for _ in range(20):
                waves.append(np.sin(2 * np.pi * freq * t) + 0.02 * rng.normal(size=t.shape))
                labels.append(label)
        spec = stft(np.asarray(waves), n_fft=64, hop_length=32)
        x = Tensor(spec)
        y = np.asarray(labels, dtype=np.int64)
        optimizer = SGD(self.model.parameters(), learning_rate=0.05)
        for _ in range(45):
            loss = cross_entropy(self.model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, waveform: np.ndarray) -> tuple[int, float]:
        values = np.asarray(waveform, dtype=np.float64)
        if values.ndim != 1 or values.size < 64:
            raise ValueError("audio waveform must contain at least 64 samples")
        values = values[:16000][None, :]
        spec = stft(values, n_fft=64, hop_length=32)
        logits = self.model(Tensor(spec)).data[0]
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(np.argmax(probs)), float(np.max(probs))


class NucleusService:
    def __init__(self) -> None:
        self.text = _TextBundle()
        self.vision = _TinyVisionBundle()
        self.audio = _TinyAudioBundle()

    def status(self) -> dict[str, object]:
        return {
            "version": "1.0.0",
            "modalities": ["text", "image", "audio"],
            "visual_learning_ready": True,
            "text_trained": self.text.model is not None,
            "vision_ready": True,
            "audio_ready": True,
            "engine": "NumPy + Nucleus autograd + visual embeddings",
        }

    def handle(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        if path == "/api/status":
            return self.status()
        if path == "/api/text/train":
            corpus = str(payload.get("corpus", ""))
            epochs = int(payload.get("epochs", 80))
            self.text.train(corpus, epochs=epochs)
            return {"ok": True, "message": "Text model trained", "status": self.status()}
        if path == "/api/text/generate":
            result = self.text.generate(
                str(payload.get("prompt", "")),
                int(payload.get("max_new_tokens", 60)),
                float(payload.get("temperature", 0.8)),
            )
            return {"text": result}
        if path == "/api/image/classify":
            pixels = np.asarray(payload.get("pixels"), dtype=np.float64)
            label, confidence = self.vision.predict(pixels)
            return {"label": "horizontal" if label == 0 else "vertical", "confidence": confidence}
        if path == "/api/audio/classify":
            waveform = np.asarray(payload.get("waveform"), dtype=np.float64)
            label, confidence = self.audio.predict(waveform)
            return {"label": "440 Hz tone" if label == 0 else "880 Hz tone", "confidence": confidence}
        raise KeyError(path)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def create_app() -> NucleusService:
    """Create the v1.0 service object."""
    return NucleusService()


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    service = create_app()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/status":
                self._send(200, _json_bytes(service.status()), "application/json")
                return
            relative = "index.html" if path in {"/", ""} else path.removeprefix("/")
            target = WEB_DIR / relative
            if not target.is_file() or WEB_DIR not in target.parents:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
                return
            content_types = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "application/javascript; charset=utf-8"}
            content_type = content_types.get(target.suffix, "text/plain; charset=utf-8")
            self._send(200, target.read_bytes(), content_type)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = service.handle(path, payload)
                self._send(200, _json_bytes(result), "application/json")
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                self._send(400, _json_bytes({"error": str(exc)}), "application/json")
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._send(500, _json_bytes({"error": f"server error: {exc}"}), "application/json")

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"[nucleus-web] {fmt % args}")

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"Project Nucleus v1.0: http://{host}:{port}")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve(host=os.getenv("NUCLEUS_HOST", "127.0.0.1"), port=int(os.getenv("NUCLEUS_PORT", "8000")))
