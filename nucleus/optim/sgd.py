"""Stochastic gradient descent optimizer."""
from nucleus.core.tensor import Tensor


class SGD:
    """Plain SGD optimizer.

    Args:
        parameters: Iterable of trainable tensors.
        learning_rate: Positive step size.
    """

    def __init__(self, parameters: list[Tensor], learning_rate: float = 0.01):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.data -= self.learning_rate * parameter.grad
