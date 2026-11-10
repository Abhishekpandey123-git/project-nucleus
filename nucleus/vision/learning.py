"""Visual representation learning and concept memory for Project Nucleus v1.0.

The v1.0 vision stack is intentionally small and NumPy-first.  It learns an
embedding from image pixels using two augmented views of the same image and a
contrastive objective, then stores learned embeddings in a lightweight visual
memory.  This is a learning/retrieval foundation for later image and video
synthesis; it is not an image generator yet.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from nucleus.core.tensor import Tensor, no_grad
from nucleus.nn.layers import Linear, Module, ReLU
from nucleus.optim.sgd import SGD


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".pgm"}


def _load_image(path: Path, image_size: int = 16) -> np.ndarray:
    """Load an image as float32 CHW in [0, 1]. Pillow is optional."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised only without Pillow
        raise RuntimeError("Pillow is required for JPEG/PNG/BMP/WebP image loading.") from exc

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float64) / 255.0
    return array.transpose(2, 0, 1)


@dataclass(frozen=True)
class VisualSample:
    path: str
    label: str


class VisualDataset:
    """Image dataset from ``root/class_name/image.*`` folders or a JSON manifest."""

    def __init__(self, samples: list[VisualSample], image_size: int = 16):
        if not samples:
            raise ValueError("visual dataset cannot be empty")
        if image_size < 4:
            raise ValueError("image_size must be at least 4")
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.labels = sorted({sample.label for sample in self.samples})
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

    @classmethod
    def from_directory(cls, root: str | Path, image_size: int = 16) -> "VisualDataset":
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {root}")
        samples: list[VisualSample] = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    samples.append(VisualSample(str(path), class_dir.name))
        return cls(samples, image_size=image_size)

    @classmethod
    def from_manifest(cls, manifest: str | Path, image_size: int = 16) -> "VisualDataset":
        """Load JSON ``[{"path": ..., "label": ...}, ...]`` samples."""
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("manifest must contain a JSON list")
        samples = []
        for item in data:
            if not isinstance(item, dict) or not item.get("path") or not item.get("label"):
                raise ValueError("each manifest item needs path and label")
            samples.append(VisualSample(str(item["path"]), str(item["label"])))
        return cls(samples, image_size=image_size)

    def load(self, indices: np.ndarray) -> np.ndarray:
        return np.stack([_load_image(Path(self.samples[int(i)].path), self.image_size) for i in indices])

    def batches(self, batch_size: int, shuffle: bool = True, seed: int | None = None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be an even integer >= 2 for contrastive training")
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.samples))
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            if len(batch) < 2:
                continue
            if len(batch) % 2:
                batch = batch[:-1]
            images = self.load(batch)
            labels = np.array([self.label_to_id[self.samples[int(i)].label] for i in batch], dtype=np.int64)
            yield images, labels


class ImageAugmenter:
    """Small deterministic-friendly image augmentations for self-supervision."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        x = np.asarray(images, dtype=np.float64).copy()
        if x.ndim != 4:
            raise ValueError("images must have shape (N,C,H,W)")
        for i in range(len(x)):
            if self.rng.random() < 0.5:
                x[i] = x[i, :, :, ::-1]
            if self.rng.random() < 0.8:
                scale = self.rng.uniform(0.85, 1.15)
                x[i] = np.clip(x[i] * scale, 0.0, 1.0)
            if self.rng.random() < 0.5:
                noise = self.rng.normal(0.0, 0.025, size=x[i].shape)
                x[i] = np.clip(x[i] + noise, 0.0, 1.0)
        return x


class TinyVisualEncoder(Module):
    """A compact trainable image encoder suitable for CPU/GPU-light experiments."""

    def __init__(self, image_size: int = 16, embedding_dim: int = 32, seed: int = 11):
        if image_size < 4 or embedding_dim < 2:
            raise ValueError("image_size and embedding_dim must be positive")
        self.image_size = int(image_size)
        self.embedding_dim = int(embedding_dim)
        input_dim = 3 * self.image_size * self.image_size
        hidden = max(64, embedding_dim * 2)
        self.proj = Linear(input_dim, hidden, seed=seed)
        self.output = Linear(hidden, embedding_dim, seed=seed + 1)
        self.relu = ReLU()

    def __call__(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                f"expected RGB NCHW images with shape (N,3,{self.image_size},{self.image_size})"
            )
        x = images.reshape(images.shape[0], -1)
        return self.output(self.relu(self.proj(x)))

    def embed(self, images: np.ndarray) -> np.ndarray:
        with no_grad():
            return self(Tensor(images)).data


class ContrastiveTrainer:
    """Train the encoder so augmented views agree and unrelated images separate."""

    def __init__(self, encoder: TinyVisualEncoder, learning_rate: float = 0.03, margin: float = 1.0):
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.encoder = encoder
        self.margin = float(margin)
        self.optimizer = SGD(encoder.parameters(), learning_rate=learning_rate, momentum=0.9)

    def step(self, images: np.ndarray, augmenter: ImageAugmenter) -> float:
        view_a = augmenter(images)
        view_b = augmenter(images)
        # Reversing the batch provides cheap negatives without labels or a large memory bank.
        negative = view_b[::-1].copy()
        za = self.encoder(Tensor(view_a, requires_grad=True))
        zb = self.encoder(Tensor(view_b, requires_grad=True))
        zn = self.encoder(Tensor(negative, requires_grad=True))
        positive_distance = ((za - zb) ** 2).mean()
        negative_distance = ((za - zn) ** 2).mean()
        separation = (self.margin - negative_distance).relu() ** 2
        loss = positive_distance + separation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.data)

    def fit(self, dataset: VisualDataset, epochs: int = 5, batch_size: int = 8, seed: int = 0) -> list[float]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        augmenter = ImageAugmenter(seed=seed)
        history: list[float] = []
        for epoch in range(epochs):
            losses = [self.step(images, augmenter) for images, _ in dataset.batches(batch_size, True, seed + epoch)]
            if not losses:
                raise ValueError("dataset does not contain enough images for training")
            history.append(float(np.mean(losses)))
        return history


class VisualMemory:
    """Persistent concept memory built from learned embeddings."""

    def __init__(self, embedding_dim: int):
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        self.embedding_dim = int(embedding_dim)
        self._embeddings: list[np.ndarray] = []
        self._labels: list[str] = []
        self._paths: list[str] = []

    def add(self, embeddings: np.ndarray, labels: list[str] | np.ndarray, paths: list[str] | None = None) -> None:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.embedding_dim:
            raise ValueError("embeddings have the wrong shape")
        labels = [str(v) for v in labels]
        if len(labels) != len(x):
            raise ValueError("one label is required for each embedding")
        if paths is None:
            paths = [""] * len(x)
        if len(paths) != len(x):
            raise ValueError("one path is required for each embedding")
        self._embeddings.extend(row.copy() for row in x)
        self._labels.extend(labels)
        self._paths.extend(str(v) for v in paths)

    @property
    def size(self) -> int:
        return len(self._labels)

    def nearest(self, embedding: np.ndarray, k: int = 5) -> list[dict[str, object]]:
        if not self._embeddings:
            return []
        query = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if query.shape != (self.embedding_dim,):
            raise ValueError("query embedding has the wrong shape")
        matrix = np.stack(self._embeddings)
        distances = np.linalg.norm(matrix - query[None, :], axis=1)
        indices = np.argsort(distances)[: max(1, int(k))]
        return [
            {"label": self._labels[i], "path": self._paths[i], "distance": float(distances[i])}
            for i in indices
        ]

    def prototypes(self) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        for embedding, label in zip(self._embeddings, self._labels):
            grouped.setdefault(label, []).append(embedding)
        return {label: np.mean(rows, axis=0) for label, rows in grouped.items()}

    def save(self, path: str | Path) -> None:
        payload = {
            "embedding_dim": self.embedding_dim,
            "embeddings": np.stack(self._embeddings) if self._embeddings else np.empty((0, self.embedding_dim)),
            "labels": np.asarray(self._labels, dtype=object),
            "paths": np.asarray(self._paths, dtype=object),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "VisualMemory":
        with np.load(path, allow_pickle=True) as data:
            memory = cls(int(data["embedding_dim"]))
            memory.add(data["embeddings"], data["labels"].tolist(), data["paths"].tolist())
            return memory


def download_manifest(manifest: str | Path, output_dir: str | Path) -> Path:
    """Download remote image URLs from a JSON manifest into a local cache.

    Manifest entries use ``{"url": "https://...", "label": "dog"}``.
    The function writes a local ``dataset.json`` that VisualDataset can consume.
    """
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("manifest must contain a JSON list")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    local_entries = []
    for index, item in enumerate(entries):
        url = str(item.get("url", ""))
        label = str(item.get("label", ""))
        if not url or not label or urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("each item needs a valid http(s) url and label")
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            suffix = ".jpg"
        destination = output / f"{index:06d}{suffix}"
        if not destination.exists():
            request = Request(url, headers={"User-Agent": "Project-Nucleus/1.0 visual-dataset"})
            with urlopen(request, timeout=20) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError(f"download exceeds 8 MiB safety limit: {url}")
            destination.write_bytes(data)
        local_entries.append({"path": str(destination), "label": label})
    dataset_manifest = output / "dataset.json"
    dataset_manifest.write_text(json.dumps(local_entries, indent=2), encoding="utf-8")
    return dataset_manifest
