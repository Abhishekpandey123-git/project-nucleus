"""Minimal neural-network modules built on :class:`nucleus.Tensor`."""
from __future__ import annotations

import numpy as np
from nucleus.core.tensor import Tensor


class Module:
    """Base class for modules with trainable parameters."""

    def parameters(self) -> list[Tensor]:
        return []


class Linear(Module):
    """Fully connected layer using ``(batch, in_features)`` inputs."""

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        rng = np.random.default_rng(seed)
        # Xavier uniform initialization for tanh-like activations.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            rng.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.weight.shape[0]:
            raise ValueError(
                f"Expected input shape (batch, {self.weight.shape[0]}), got {inputs.shape}."
            )
        return inputs @ self.weight + self.bias

    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]


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

    def parameters(self) -> list[Tensor]:
        return [parameter for module in self.modules for parameter in module.parameters()]
