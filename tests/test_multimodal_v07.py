import numpy as np
import pytest

from nucleus.multimodal.metrics import recall_at_k


def test_recall_at_k_perfect_diagonal():
    scores = np.eye(4) * 10
    result = recall_at_k(scores, ks=(1, 2))
    assert result["image_to_text_R@1"] == 1.0
    assert result["text_to_image_R@1"] == 1.0
    assert result["image_to_text_R@2"] == 1.0


def test_recall_at_k_worst_case():
    scores = np.array([
        [0.0, 3.0, 2.0],
        [3.0, 0.0, 2.0],
        [3.0, 2.0, 0.0],
    ])
    result = recall_at_k(scores, ks=(1, 3))
    assert result["image_to_text_R@3"] == 1.0
    assert result["text_to_image_R@3"] == 1.0


def test_recall_rejects_non_square_matrix():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((2, 3)))


def test_recall_rejects_empty_matrix():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((0, 0)))


def test_cosine_similarity_matrix_shape_and_diagonal():
    from nucleus.core.tensor import Tensor
    from nucleus.multimodal.loss import cosine_similarity_matrix

    images = Tensor(np.eye(3), requires_grad=True)
    texts = Tensor(np.eye(3), requires_grad=True)

    similarities = cosine_similarity_matrix(images, texts)

    assert similarities.shape == (3, 3)
    np.testing.assert_allclose(similarities.data, np.eye(3), atol=1e-6)


def test_contrastive_loss_is_finite_scalar():
    from nucleus.core.tensor import Tensor
    from nucleus.multimodal.loss import symmetric_contrastive_loss

    images = Tensor(
        np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        requires_grad=True,
    )
    texts = Tensor(
        np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64),
        requires_grad=True,
    )

    loss = symmetric_contrastive_loss(images, texts)

    assert loss.shape == ()
    assert np.isfinite(loss.data).all()
    assert loss.data.item() >= 0


def test_contrastive_loss_backpropagates_finite_gradients():
    from nucleus.core.tensor import Tensor
    from nucleus.multimodal.loss import symmetric_contrastive_loss

    images = Tensor(
        np.array([[0.9, 0.1, 0.0],
                  [0.1, 0.8, 0.1],
                  [0.0, 0.2, 0.9]], dtype=np.float64),
        requires_grad=True,
    )
    texts = Tensor(
        np.array([[0.8, 0.2, 0.1],
                  [0.2, 0.7, 0.2],
                  [0.1, 0.3, 0.8]], dtype=np.float64),
        requires_grad=True,
    )

    loss = symmetric_contrastive_loss(images, texts)
    loss.backward()

    assert images.grad is not None
    assert texts.grad is not None
    assert np.isfinite(images.grad).all()
    assert np.isfinite(texts.grad).all()
    assert np.linalg.norm(images.grad) > 0
    assert np.linalg.norm(texts.grad) > 0


@pytest.mark.parametrize("temperature", [0, -0.1, float("inf"), float("nan")])
def test_contrastive_loss_rejects_invalid_temperature(temperature):
    from nucleus.core.tensor import Tensor
    from nucleus.multimodal.loss import symmetric_contrastive_loss

    embeddings = Tensor(np.eye(2), requires_grad=True)

    with pytest.raises(ValueError):
        symmetric_contrastive_loss(embeddings, embeddings, temperature)


def test_image_encoder_forward_and_backward():
    from nucleus.core.tensor import Tensor
    from nucleus.multimodal import ImageEncoder

    rng = np.random.default_rng(10)
    encoder = ImageEncoder(in_channels=1, feature_dim=16)
    images = Tensor(rng.normal(size=(2, 1, 12, 12)))

    features = encoder(images)
    assert features.shape == (2, 16)

    features.sum().backward()

    params = encoder.parameters()
    assert params
    assert all(p.grad is not None for p in params)
    assert all(np.isfinite(p.grad).all() for p in params)


def test_text_encoder_forward_and_backward():
    from nucleus.multimodal import TextEncoder

    encoder = TextEncoder(vocab_size=30, embedding_dim=32)
    token_ids = np.array([[1, 2, 3], [4, 5, 6]])

    features = encoder(token_ids)
    assert features.shape == (2, 32)

    features.sum().backward()

    params = encoder.parameters()
    assert params
    assert all(p.grad is not None for p in params)
    assert all(np.isfinite(p.grad).all() for p in params)


def test_combined_model_forward_and_backward():
    from nucleus.core.tensor import Tensor
    from nucleus.multimodal import (
        ImageEncoder,
        ImageTextAlignmentModel,
        TextEncoder,
        symmetric_contrastive_loss,
    )
    from nucleus.nn.layers import Linear

    rng = np.random.default_rng(42)

    model = ImageTextAlignmentModel(
        ImageEncoder(in_channels=1, feature_dim=16),
        TextEncoder(vocab_size=30, embedding_dim=32),
        Linear(16, 8, seed=21),
        Linear(32, 8, seed=22),
    )

    images = Tensor(rng.normal(size=(2, 1, 12, 12)))
    token_ids = rng.integers(0, 30, size=(2, 5))

    image_features, text_features = model(images, token_ids)

    assert image_features.shape == (2, 8)
    assert text_features.shape == (2, 8)

    loss = symmetric_contrastive_loss(image_features, text_features)
    loss.backward()

    params = model.parameters()
    assert len(params) == 9
    assert all(p.grad is not None for p in params)
    assert all(np.isfinite(p.grad).all() for p in params)

@pytest.mark.parametrize(
    "scores",
    [
        np.array([[1.0, np.nan], [0.2, 1.0]]),
        np.array([[1.0, np.inf], [0.2, 1.0]]),
        np.array([[1.0, -np.inf], [0.2, 1.0]]),
    ],
)
def test_recall_rejects_non_finite_scores(scores):
    with pytest.raises(ValueError, match="finite"):
        recall_at_k(scores)


@pytest.mark.parametrize("ks", [(), [], (0,), (-1,), (1.5,), (True,)])
def test_recall_rejects_invalid_cutoffs(ks):
    with pytest.raises(ValueError):
        recall_at_k(np.eye(3), ks=ks)


def test_recall_rejects_non_iterable_cutoffs():
    with pytest.raises(ValueError):
        recall_at_k(np.eye(3), ks=5)


def test_recall_accepts_numpy_integer_cutoffs():
    result = recall_at_k(np.eye(3), ks=(np.int64(1), np.int64(2)))
    assert result["image_to_text_R@1"] == 1.0
    assert result["text_to_image_R@2"] == 1.0