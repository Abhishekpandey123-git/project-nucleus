# Project Nucleus v1.0 — Visual Learning Engine

Project Nucleus v1.0 adds a small, from-scratch visual representation learner to the NumPy neural engine.

## What changed

v1.0 moves beyond fixed image classification. Nucleus can now learn an embedding for images from augmented views, keep those learned representations in a visual memory, retrieve similar examples, and calculate concept prototypes.

```text
Internet / local images
          ↓
     VisualDataset
          ↓
    image augmentation
          ↓
    TinyVisualEncoder
          ↓
   learned embedding
          ↓
     VisualMemory
      ↙         ↘
 nearest       prototypes
 examples       per concept
```

### Dataset layout

```text
objects/
├── cat/
│   ├── cat01.jpg
│   └── cat02.jpg
├── dog/
│   ├── dog01.jpg
│   └── dog02.jpg
└── car/
    └── car01.jpg
```

### Train

```bash
python -m pip install -e ".[dev,vision]"
python examples/visual_learning_v10.py --data ./objects --epochs 5 --batch-size 8
```

The command creates a compressed `nucleus_visual_memory_v10.npz` containing learned image embeddings and source metadata.

### Internet images

Create a JSON manifest:

```json
[
  {"url": "https://example.org/cat.jpg", "label": "cat"},
  {"url": "https://example.org/dog.jpg", "label": "dog"}
]
```

Download into a local cache:

```python
from nucleus.vision import download_manifest
manifest = download_manifest("urls.json", "data/internet-cache")
```

Then load `manifest` with `VisualDataset.from_manifest(...)`.

Only download images you are allowed to use. v1.0 intentionally requires an explicit URL manifest rather than silently scraping arbitrary websites.

## Current scope

v1.0 is a **visual learning foundation**, not a generative model. It can learn and retrieve visual representations, but it cannot yet create a new cat/person/object image or generate video. Those are the next milestones.
