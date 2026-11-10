"""Project Nucleus v1.0 visual concept learning demo.

Usage:
  python examples/visual_learning_v10.py --data ./data/objects --epochs 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure this example always imports the local Project Nucleus package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nucleus.vision import ContrastiveTrainer, TinyVisualEncoder, VisualDataset, VisualMemory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="directory containing class/image files")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--memory", default="nucleus_visual_memory_v10.npz")
    args = parser.parse_args()

    dataset = VisualDataset.from_directory(args.data, image_size=args.image_size)
    encoder = TinyVisualEncoder(args.image_size, args.embedding_dim)
    trainer = ContrastiveTrainer(encoder)
    history = trainer.fit(dataset, epochs=args.epochs, batch_size=args.batch_size)

    memory = VisualMemory(args.embedding_dim)
    for start in range(0, len(dataset.samples), args.batch_size):
        end = min(start + args.batch_size, len(dataset.samples))
        indices = list(range(start, end))
        images = dataset.load(__import__("numpy").array(indices, dtype=int))
        embeddings = encoder.embed(images)
        labels = [dataset.samples[i].label for i in indices]
        paths = [dataset.samples[i].path for i in indices]
        memory.add(embeddings, labels, paths)
    memory.save(args.memory)

    print(json.dumps({
        "version": "1.0.0",
        "samples": len(dataset.samples),
        "concepts": dataset.labels,
        "embedding_dim": args.embedding_dim,
        "loss": history,
        "memory": str(Path(args.memory).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
