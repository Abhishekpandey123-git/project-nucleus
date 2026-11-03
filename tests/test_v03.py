import numpy as np
import pytest

from nucleus import Parameter, Tensor, no_grad
from nucleus.nn import Linear, Module, ReLU, Sequential, Tanh
from nucleus.optim import SGD


def test_parameter_defaults_to_trainable_tensor():
    parameter = Parameter([[1.0, 2.0]])
    assert isinstance(parameter, Tensor)
    assert parameter.requires_grad
    np.testing.assert_array_equal(parameter.grad, [[0.0, 0.0]])


def test_module_recursively_discovers_parameters_once():
    model = Sequential(Linear(2, 3, seed=1), Tanh(), Linear(3, 1, seed=2))
    params = model.parameters()
    assert len(params) == 4
    assert len({id(p) for p in params}) == 4
    assert all(isinstance(p, Parameter) for p in params)


def test_module_discovers_parameters_in_nested_containers():
    class Container(Module):
        def __init__(self):
            self.blocks = [Linear(2, 2, seed=0), {"head": Linear(2, 1, seed=1)}]
    model = Container()
    assert len(model.parameters()) == 4


def test_no_grad_disables_graph_and_restores_state():
    x = Parameter([1.0, 2.0])
    with no_grad():
        y = (x * 3.0).tanh()
        assert not y.requires_grad
    z = (x * 3.0).sum()
    assert z.requires_grad
    z.backward()
    np.testing.assert_array_equal(x.grad, [3.0, 3.0])


def test_no_grad_restores_after_exception():
    x = Parameter([2.0])
    with pytest.raises(RuntimeError):
        with no_grad():
            raise RuntimeError("test")
    assert (x * 2).requires_grad


def test_backward_accumulates_gradients_until_zeroed():
    x = Parameter([2.0, -1.0])
    (x * x).sum().backward()
    (3.0 * x).sum().backward()
    np.testing.assert_allclose(x.grad, [7.0, 1.0])
    x.zero_grad()
    np.testing.assert_array_equal(x.grad, [0.0, 0.0])


def test_sgd_momentum_and_state_round_trip():
    p1 = Parameter([1.0, 2.0])
    opt1 = SGD([p1], learning_rate=0.1, momentum=0.9)
    p1.grad[...] = [2.0, -1.0]
    opt1.step()
    state = opt1.state_dict()

    p2 = Parameter(p1.data.copy())
    opt2 = SGD([p2], learning_rate=0.5, momentum=0.0)
    opt2.load_state_dict(state)
    assert opt2.learning_rate == pytest.approx(0.1)
    assert opt2.momentum == pytest.approx(0.9)
    p1.grad[...] = [1.0, 1.0]
    p2.grad[...] = [1.0, 1.0]
    opt1.step()
    opt2.step()
    np.testing.assert_allclose(p2.data, p1.data)


def test_optimizer_state_rejects_wrong_parameter_shapes():
    source = Parameter([1.0, 2.0])
    state = SGD([source]).state_dict()
    target = Parameter([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shapes"):
        SGD([target]).load_state_dict(state)


def test_sgd_validates_hyperparameters():
    with pytest.raises(ValueError):
        SGD([Parameter([1.0])], learning_rate=0)
    with pytest.raises(ValueError):
        SGD([Parameter([1.0])], momentum=1.0)


def test_sequential_training_with_parameter_discovery():
    model = Sequential(Linear(2, 4, seed=3), ReLU(), Linear(4, 1, seed=4))
    optimizer = SGD(model.parameters(), learning_rate=0.05, momentum=0.8)
    inputs = Tensor([[0.0, 1.0], [1.0, 0.0]])
    before = [p.data.copy() for p in model.parameters()]
    prediction = model(inputs)
    loss = ((prediction - Tensor([[1.0], [1.0]])) ** 2).mean()
    loss.backward()
    optimizer.step()
    assert any(not np.array_equal(old, new.data) for old, new in zip(before, model.parameters()))
