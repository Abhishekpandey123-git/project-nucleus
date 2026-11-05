# Integrating the v0.8 add-on

This ZIP contains the v0.8 audio feature files, not a duplicate copy of the
entire Git repository.

1. Open your Project Nucleus repository (the v0.7 codebase).
2. Copy the `nucleus/audio` directory into the existing `nucleus/` directory.
3. Copy `examples/audio_classification_demo.py` into `examples/`.
4. Copy `tests/test_audio_v08.py` into `tests/`.
5. Copy `docs/v0.8-audio-learning.md` into `docs/`.
6. Copy `README_V08.md` and `V0.8_CHANGELOG.md` into the repository root.
7. Run the commands below.

```bash
python examples/audio_classification_demo.py --epochs 25
pytest -q tests/test_audio_v08.py
python -m ruff check nucleus/audio examples/audio_classification_demo.py tests/test_audio_v08.py
```

If your optimizer is stored at a different import path, adjust the demo import
from `nucleus.optim`. Do not replace existing files from v0.7.
