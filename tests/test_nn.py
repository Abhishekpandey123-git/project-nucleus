import numpy as np
import pytest

from nucleus import Tensor
from nucleus.nn import Linear, Sequential, Tanh
from nucleus.nn.losses import mean_squared_error
from nucleus.optim import SGD


def test_linear_shapes_and_parameter_count():
    layer = Linear(3, 2, seed=7)
    output = layer(Tensor(np.ones((4, 3))))
    assert output.shape == (4, 2)
    assert len(layer.parameters()) == 2


def test_xor_tiny_network_learns():
    # Deterministic tiny sanity experiment, not a benchmark claim.
    x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = Tensor([[0], [1], [1], [0]])
    model = Sequential(Linear(2, 4, seed=11), Tanh(), Linear(4, 1, seed=12))
    optimizer = SGD(model.parameters(), learning_rate=0.08)

    initial = float(mean_squared_error(model(x), y).data)
    for _ in range(1200):
        predictions = model(x)
        loss = mean_squared_error(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final = float(mean_squared_error(model(x), y).data)
    assert final < initial
    assert final < 0.08


def test_mse_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        mean_squared_error(Tensor([[1.0]]), Tensor([1.0]))
