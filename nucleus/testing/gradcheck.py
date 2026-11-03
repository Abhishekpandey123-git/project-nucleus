"""Numerical gradient checking for Tensor reverse-mode autodiff."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from nucleus.core.tensor import Tensor

@dataclass(frozen=True)
class GradientCheckResult:
    analytical: np.ndarray
    numerical: np.ndarray
    max_absolute_error: float
    max_relative_error: float
    passed: bool
    eps: float
    atol: float
    rtol: float

def _scalar_value(function: Callable[[Tensor], Tensor], data: np.ndarray) -> float:
    output = function(Tensor(data, requires_grad=False))
    if not isinstance(output, Tensor):
        raise TypeError("The function passed to gradcheck must return a Tensor.")
    if output.data.size != 1:
        raise ValueError("gradcheck requires a scalar Tensor output. Reduce it with .sum() or .mean().")
    value = float(output.data.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("The function produced a non-finite scalar output.")
    return value

def gradcheck(function: Callable[[Tensor], Tensor], input: Tensor | object, *, eps: float = 1e-6, atol: float = 1e-5, rtol: float = 1e-4) -> GradientCheckResult:
    if not callable(function):
        raise TypeError("function must be callable.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number.")
    if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("atol and rtol must be finite non-negative numbers.")
    original = input.data.copy() if isinstance(input, Tensor) else np.asarray(input, dtype=np.float64).copy()
    if not np.all(np.isfinite(original)):
        raise ValueError("input must contain only finite values.")
    original = original.astype(np.float64, copy=True)
    x = Tensor(original, requires_grad=True)
    output = function(x)
    if not isinstance(output, Tensor):
        raise TypeError("The function passed to gradcheck must return a Tensor.")
    if output.data.size != 1:
        raise ValueError("gradcheck requires a scalar Tensor output. Reduce it with .sum() or .mean().")
    if not np.all(np.isfinite(output.data)):
        raise ValueError("The function produced a non-finite scalar output.")
    output.backward()
    analytical = x.grad.copy()
    numerical = np.zeros_like(original)
    for index in np.ndindex(original.shape):
        plus, minus = original.copy(), original.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical[index] = (_scalar_value(function, plus) - _scalar_value(function, minus)) / (2 * eps)
    difference = np.abs(analytical - numerical)
    denominator = np.maximum(np.maximum(np.abs(analytical), np.abs(numerical)), np.finfo(float).tiny)
    return GradientCheckResult(analytical, numerical,
        float(np.max(difference)) if difference.size else 0.0,
        float(np.max(difference / denominator)) if difference.size else 0.0,
        bool(np.allclose(analytical, numerical, atol=atol, rtol=rtol)), float(eps), float(atol), float(rtol))
