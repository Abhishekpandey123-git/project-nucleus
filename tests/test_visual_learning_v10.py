import numpy as np

from nucleus.vision import ContrastiveTrainer, TinyVisualEncoder, VisualMemory


def test_visual_encoder_forward_and_backward():
    rng = np.random.default_rng(3)
    images = rng.random((4, 3, 8, 8))
    encoder = TinyVisualEncoder(image_size=8, embedding_dim=12)
    trainer = ContrastiveTrainer(encoder, learning_rate=0.01)
    loss = trainer.step(images, __import__("nucleus.vision", fromlist=["ImageAugmenter"]).ImageAugmenter(2))
    assert np.isfinite(loss)
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in encoder.parameters())


def test_visual_memory_nearest_and_roundtrip(tmp_path):
    memory = VisualMemory(3)
    memory.add(np.array([[1., 0., 0.], [0., 1., 0.]]), ["cat", "dog"], ["cat.jpg", "dog.jpg"])
    result = memory.nearest(np.array([0.9, 0.1, 0.0]), k=1)
    assert result[0]["label"] == "cat"
    path = tmp_path / "memory.npz"
    memory.save(path)
    restored = VisualMemory.load(path)
    assert restored.size == 2
    assert restored.nearest(np.array([0.0, 0.9, 0.1]), 1)[0]["label"] == "dog"
