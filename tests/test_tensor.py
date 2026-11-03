import numpy as np
import pytest

from nucleus import Tensor


def test_scalar_chain_rule():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + 2 * x
    y.backward()
    assert y.data == pytest.approx(15.0)
    assert x.grad == pytest.approx(8.0)


def test_broadcast_gradient_is_reduced():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = Tensor([[2.0, 3.0]], requires_grad=True)
    (x + bias).sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 2)))
    np.testing.assert_allclose(bias.grad, [[2.0, 2.0]])


def test_matmul_gradients():
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)
    (x @ w).sum().backward()
    np.testing.assert_allclose(x.grad, [[3.0, 4.0]])
    np.testing.assert_allclose(w.grad, [[1.0], [2.0]])


def test_mean_and_tanh_gradient():
    x = Tensor([0.2, -0.4], requires_grad=True)
    x.tanh().mean().backward()
    expected = (1 - np.tanh(x.data) ** 2) / 2
    np.testing.assert_allclose(x.grad, expected)


def test_backward_requires_seed_for_non_scalar():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()
    x.backward([1.0, 1.0])
    np.testing.assert_allclose(x.grad, [1.0, 1.0])


def test_log_domain_validation():
    with pytest.raises(ValueError):
        Tensor([0.0]).log()
