import numpy as np
import pytest
from nucleus import Tensor
from nucleus.nn.attention import MultiHeadAttention, scaled_dot_product_attention, softmax
from nucleus.nn.layers import Linear
from nucleus.nn.transformer import LayerNorm, PositionalEncoding, TransformerBlock
from nucleus.optim import SGD


def test_batched_matmul_and_gradient():
    a = Tensor(np.arange(24.).reshape(2, 3, 4), True)
    b = Tensor(np.arange(40.).reshape(2, 4, 5), True)
    y = a @ b
    assert y.shape == (2, 3, 5)
    y.sum().backward()
    np.testing.assert_allclose(a.grad, np.matmul(np.ones((2, 3, 5)), b.data.swapaxes(-1, -2)))
    np.testing.assert_allclose(b.grad, np.matmul(a.data.swapaxes(-1, -2), np.ones((2, 3, 5))))


def test_batched_matmul_broadcast_gradient():
    a = Tensor(np.ones((2, 3, 4)), True)
    b = Tensor(np.ones((1, 4, 5)), True)
    (a @ b).sum().backward()
    np.testing.assert_allclose(b.grad, np.full((1, 4, 5), 6.))


def test_reshape_transpose_backward():
    x = Tensor(np.arange(24.).reshape(2, 3, 4), True)
    x.transpose(0, 2, 1).reshape(2, 12).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones_like(x.data))


def test_linear_sequence_input():
    assert Linear(4, 6)(Tensor(np.ones((2, 3, 4)))).shape == (2, 3, 6)


def test_softmax_backward():
    x = Tensor(np.array([[[1., 2., 3.]]]), True)
    p = softmax(x)
    np.testing.assert_allclose(p.data.sum(-1), 1.)
    p.sum().backward()
    assert np.isfinite(x.grad).all()


def test_causal_attention_mask():
    q = Tensor(np.ones((1, 1, 3, 2)))
    k = Tensor(np.ones((1, 1, 3, 2)))
    v = Tensor(np.array([[[[1., 0.], [0., 1.], [9., 9.]]]]))
    out, weights = scaled_dot_product_attention(q, k, v, causal=True)
    np.testing.assert_allclose(weights.data[0, 0, 0], [1., 0., 0.])
    assert weights.data[0, 0, 1, 2] == pytest.approx(0.)
    np.testing.assert_allclose(out.data[0, 0, 0], [1., 0.])


def test_multihead_forward_backward():
    layer = MultiHeadAttention(8, 2, seed=3)
    x = Tensor(np.random.default_rng(1).normal(size=(2, 4, 8)), True)
    y = layer(x, causal=True)
    assert y.shape == (2, 4, 8)
    y.sum().backward()
    assert np.isfinite(x.grad).all()
    assert all(p.grad is not None for p in layer.parameters())


def test_invalid_head_count():
    with pytest.raises(ValueError):
        MultiHeadAttention(7, 2)


def test_positional_encoding_layer_norm():
    encoded = PositionalEncoding(6, 8)(Tensor(np.zeros((1, 4, 6))))
    assert not np.array_equal(encoded.data[:, 0], encoded.data[:, 1])
    norm = LayerNorm(6)
    x = Tensor(np.array([[[1., 2., 3., 4., 5., 6.]]]), True)
    y = norm(x)
    np.testing.assert_allclose(y.data.mean(-1), 0., atol=1e-6)
    y.sum().backward()
    assert norm.gamma.grad is not None


def test_transformer_block_optimizer_update():
    block = TransformerBlock(8, 2, 16, seed=5)
    x = Tensor(np.random.default_rng(2).normal(size=(2, 4, 8)), True)
    y = block(x)
    loss = (y ** 2).mean()
    opt = SGD(block.parameters(), learning_rate=.01)
    opt.zero_grad()
    loss.backward()
    before = [p.data.copy() for p in block.parameters()]
    opt.step()
    assert any(not np.array_equal(a, p.data) for a, p in zip(before, block.parameters()))
