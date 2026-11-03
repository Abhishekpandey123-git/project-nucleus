import numpy as np
import pytest

from nucleus import Tensor
from nucleus.testing import gradcheck


@pytest.mark.parametrize(
    "function, values",
    [
        (lambda x: (x + 2.3).sum(), [0.2, -0.7, 1.4]),
        (lambda x: (x * x).sum(), [0.2, -0.7, 1.4]),
        (lambda x: (x / 2.5).sum(), [0.2, -0.7, 1.4]),
        (lambda x: (x ** 3).sum(), [0.2, -0.7, 1.4]),
        (lambda x: x.relu().sum(), [0.2, -0.7, 1.4]),
        (lambda x: x.sigmoid().sum(), [0.2, -0.7, 1.4]),
        (lambda x: x.tanh().sum(), [0.2, -0.7, 1.4]),
        (lambda x: x.exp().sum(), [-0.4, 0.1, 0.7]),
        (lambda x: x.log().sum(), [0.4, 1.2, 2.0]),
        (lambda x: x.mean(), [0.2, -0.7, 1.4]),
        (lambda x: x.sum(), [0.2, -0.7, 1.4]),
    ],
)
def test_gradcheck_supported_unary_and_elementwise_ops(function, values):
    result = gradcheck(function, Tensor(values))
    assert result.passed, result
    np.testing.assert_allclose(result.analytical, result.numerical, atol=1e-5, rtol=1e-4)


def test_gradcheck_broadcast_addition():
    bias = Tensor([[0.3, -0.2]])
    result = gradcheck(lambda x: (x + bias).sum(), [[0.1, 0.5], [-0.4, 0.8]])
    assert result.passed


def test_gradcheck_matrix_multiplication():
    weight = Tensor([[0.2, -0.3], [0.7, 0.4]])
    result = gradcheck(lambda x: ((x @ weight) ** 2).sum(), [[0.5, -0.6], [0.1, 0.9]])
    assert result.passed, result


def test_gradcheck_combined_expression():
    result = gradcheck(
        lambda x: ((x * x + 2.0 * x) / 3.0).tanh().sum(),
        [-0.8, 0.2, 0.9],
    )
    assert result.passed, result


def test_gradcheck_does_not_mutate_tensor_input():
    x = Tensor([0.2, 0.5])
    before = x.data.copy()
    gradcheck(lambda value: (value * value).sum(), x)
    np.testing.assert_array_equal(x.data, before)


def test_gradcheck_requires_scalar_output():
    with pytest.raises(ValueError, match="scalar"):
        gradcheck(lambda x: x * x, [1.0, 2.0])


def test_gradcheck_rejects_invalid_epsilon():
    with pytest.raises(ValueError, match="eps"):
        gradcheck(lambda x: x.sum(), [1.0], eps=0)


def test_gradcheck_reports_mismatch():
    # Deliberately corrupt the expected derivative by using a custom function
    # whose backward graph does not represent its numerical forward value.
    def incorrect(x):
        y = x * 2.0
        y.data[...] = x.data ** 2
        return y.sum()

    result = gradcheck(incorrect, [0.5, 1.5])
    assert not result.passed
