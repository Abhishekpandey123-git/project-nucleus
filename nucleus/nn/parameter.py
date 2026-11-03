"""Trainable parameter type."""
from nucleus.core.tensor import Tensor


class Parameter(Tensor):
    """A Tensor that represents a trainable model parameter by default."""

    def __init__(self, data: object, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)
