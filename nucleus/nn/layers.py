"""Neural-network modules with recursive parameter discovery."""
from __future__ import annotations

import numpy as np
from nucleus.core.tensor import Tensor
from nucleus.nn.parameter import Parameter


class Module:
    """Base class that discovers Parameters in nested modules and containers."""

    def parameters(self) -> list[Parameter]:
        found: list[Parameter] = []
        seen: set[int] = set()

        def visit(value: object) -> None:
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
            elif isinstance(value, Module):
                for child in vars(value).values():
                    visit(child)
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        for value in vars(self).values():
            visit(value)
        return found


class Linear(Module):
    """Fully connected layer using ``(batch, in_features)`` inputs."""

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(rng.uniform(-limit, limit, (in_features, out_features)))
        self.bias = Parameter(np.zeros((1, out_features)))

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 2 or inputs.shape[-1] != self.weight.shape[0]:
            raise ValueError(
                f"Expected input with final dimension {self.weight.shape[0]}, "
                f"got {inputs.shape}."
            )
        return inputs @ self.weight + self.bias


class ReLU(Module):
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class Tanh(Module):
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()


class Sigmoid(Module):
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.sigmoid()


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = list(modules)

    def __call__(self, inputs: Tensor) -> Tensor:
        for module in self.modules:
            inputs = module(inputs)
        return inputs
