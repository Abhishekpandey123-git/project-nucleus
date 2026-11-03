"""A small NumPy-backed Tensor with reverse-mode automatic differentiation."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator
import numpy as np


_GRAD_ENABLED: ContextVar[bool] = ContextVar("nucleus_grad_enabled", default=True)


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable graph construction for inference/evaluation.

    Existing leaf tensors retain their ``requires_grad`` setting; operations
    performed inside this context produce tensors that do not track gradients.
    The context is safely restored even if an exception is raised.
    """
    token = _GRAD_ENABLED.set(False)
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(token)


def is_grad_enabled() -> bool:
    """Return whether operations currently record autograd history."""
    return _GRAD_ENABLED.get()


def _operation_requires_grad(*tensors: Tensor) -> bool:
    return is_grad_enabled() and any(t.requires_grad for t in tensors)


def _as_array(value: object) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.data
    return np.asarray(value, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A differentiable NumPy array."""

    def __init__(self, data: object, requires_grad: bool = False):
        self.data = _as_array(data).astype(np.float64, copy=True)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = np.zeros_like(self.data) if self.requires_grad else None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Callable[[], None] = lambda: None
        self._op = "leaf"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, gradient: object | None = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor without gradients.")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("gradient is required when backward() is called on a non-scalar tensor.")
            seed = np.ones_like(self.data)
        else:
            seed = _as_array(gradient)
            if seed.shape != self.shape:
                raise ValueError(f"gradient shape {seed.shape} does not match tensor shape {self.shape}")

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)
        assert self.grad is not None
        self.grad += seed
        for node in reversed(topo):
            node._backward()

    def _binary(self, other: object, op: str) -> Tensor:
        rhs = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self.data, rhs.data
        requires = _operation_requires_grad(self, rhs)
        if op == "add":
            out_data = a + b
        elif op == "mul":
            out_data = a * b
        elif op == "div":
            out_data = a / b
        else:
            raise ValueError(f"Unsupported binary operation: {op}")
        out = Tensor(out_data, requires)
        out._parents = (self, rhs) if requires else ()
        out._op = op

        def backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                local = out.grad if op == "add" else (out.grad * b if op == "mul" else out.grad / b)
                self.grad += _unbroadcast(local, self.shape)
            if rhs.requires_grad:
                assert rhs.grad is not None
                local = out.grad if op == "add" else (out.grad * a if op == "mul" else -out.grad * a / (b ** 2))
                rhs.grad += _unbroadcast(local, rhs.shape)

        if requires:
            out._backward = backward
        return out

    def __add__(self, other: object) -> Tensor:
        return self._binary(other, "add")

    def __radd__(self, other: object) -> Tensor:
        return self + other

    def __mul__(self, other: object) -> Tensor:
        return self._binary(other, "mul")

    def __rmul__(self, other: object) -> Tensor:
        return self * other

    def __truediv__(self, other: object) -> Tensor:
        return self._binary(other, "div")

    def __rtruediv__(self, other: object) -> Tensor:
        return Tensor(other) / self

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: object) -> Tensor:
        return self + (-other if isinstance(other, Tensor) else -_as_array(other))

    def __rsub__(self, other: object) -> Tensor:
        return Tensor(other) - self

    def __pow__(self, power: float) -> Tensor:
        if not isinstance(power, (int, float)):
            raise TypeError("Tensor power currently supports numeric scalar exponents.")
        requires = _operation_requires_grad(self)
        out = Tensor(self.data ** power, requires)
        out._parents = (self,) if requires else ()
        out._op = "pow"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad * power * (self.data ** (power - 1))

        if requires:
            out._backward = backward
        return out


    def __matmul__(self, other: object) -> Tensor:
        rhs = other if isinstance(other, Tensor) else Tensor(other)
        if self.ndim < 2 or rhs.ndim < 2:
            raise ValueError("matmul requires operands with at least 2 dimensions.")
        if self.shape[-1] != rhs.shape[-2]:
            raise ValueError(f"Incompatible matrix shapes: {self.shape} and {rhs.shape}")
        requires = _operation_requires_grad(self, rhs)
        out = Tensor(np.matmul(self.data, rhs.data), requires)
        out._parents = (self, rhs) if requires else ()
        out._op = "matmul"

        def backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += _unbroadcast(np.matmul(out.grad, np.swapaxes(rhs.data, -1, -2)), self.shape)
            if rhs.requires_grad:
                assert rhs.grad is not None
                rhs.grad += _unbroadcast(np.matmul(np.swapaxes(self.data, -1, -2), out.grad), rhs.shape)

        if requires:
            out._backward = backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        requires = _operation_requires_grad(self)
        out = Tensor(self.data.reshape(tuple(shape)), requires)
        out._parents = (self,) if requires else ()
        out._op = "reshape"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad.reshape(self.shape)

        if requires:
            out._backward = backward
        return out

    def transpose(self, *axes: int) -> Tensor:
        permutation = tuple(axes) if axes else tuple(reversed(range(self.ndim)))
        if sorted(permutation) != list(range(self.ndim)):
            raise ValueError("axes must be a permutation of all tensor dimensions.")
        inverse = tuple(np.argsort(permutation))
        requires = _operation_requires_grad(self)
        out = Tensor(self.data.transpose(permutation), requires)
        out._parents = (self,) if requires else ()
        out._op = "transpose"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad.transpose(inverse)

        if requires:
            out._backward = backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        requires = _operation_requires_grad(self)
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires)
        out._parents = (self,) if requires else ()
        out._op = "sum"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted((a % self.ndim for a in axes)):
                        grad = np.expand_dims(grad, ax)
                self.grad += np.broadcast_to(grad, self.shape)

        if requires:
            out._backward = backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def exp(self) -> Tensor:
        values = np.exp(self.data)
        requires = _operation_requires_grad(self)
        out = Tensor(values, requires)
        out._parents = (self,) if requires else ()
        out._op = "exp"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad * values

        if requires:
            out._backward = backward
        return out

    def log(self) -> Tensor:
        if np.any(self.data <= 0):
            raise ValueError("log() requires strictly positive input values.")
        requires = _operation_requires_grad(self)
        out = Tensor(np.log(self.data), requires)
        out._parents = (self,) if requires else ()
        out._op = "log"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad / self.data

        if requires:
            out._backward = backward
        return out

    def relu(self) -> Tensor:
        values = np.maximum(0.0, self.data)
        requires = _operation_requires_grad(self)
        out = Tensor(values, requires)
        out._parents = (self,) if requires else ()
        out._op = "relu"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad * (self.data > 0)

        if requires:
            out._backward = backward
        return out

    def tanh(self) -> Tensor:
        values = np.tanh(self.data)
        requires = _operation_requires_grad(self)
        out = Tensor(values, requires)
        out._parents = (self,) if requires else ()
        out._op = "tanh"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad * (1.0 - values ** 2)

        if requires:
            out._backward = backward
        return out

    def sigmoid(self) -> Tensor:
        values = np.empty_like(self.data)
        positive = self.data >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        values[~positive] = exp_x / (1.0 + exp_x)
        requires = _operation_requires_grad(self)
        out = Tensor(values, requires)
        out._parents = (self,) if requires else ()
        out._op = "sigmoid"

        def backward() -> None:
            if self.requires_grad and out.grad is not None:
                assert self.grad is not None
                self.grad += out.grad * values * (1.0 - values)

        if requires:
            out._backward = backward
        return out

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"
