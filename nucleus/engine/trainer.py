"""Small, explicit training-step helper."""
from nucleus.core.tensor import Tensor
from nucleus.optim.sgd import SGD


def train_step(loss: Tensor, optimizer: SGD) -> float:
    """Backpropagate one scalar loss, update parameters, and return its value."""
    if loss.data.size != 1:
        raise ValueError("train_step expects a scalar loss.")
    value = float(loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return value
