"""Numerical gradient checking for Tensor reverse-mode autodiff."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from nucleus.core.tensor import Tensor


@dataclass(frozen=True)
class GradientCheckResult:
    """Comparison between analytical and finite-difference gradients."""

    analytical: np.ndarray
    numerical: np.ndarray
    max_absolute_error: float
    max_relative_error: float
    passed: bool
    eps: float
    atol: float
    rtol: float


def _scalar_value(function: Callable[[Tensor], Tensor], data: np.ndarray) -> float:
    """Evaluate a function without recording gradients; require scalar output."""
    output = function(Tensor(data, requires_grad=False))
    if not isinstance(output, Tensor):
        raise TypeError("The function passed to gradcheck must return a Tensor.")
    if output.data.size != 1:
        raise ValueError(
            "gradcheck requires a scalar Tensor output. Reduce the output, "
            "for example with .sum() or .mean()."
        )
    value = float(output.data.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("The function produced a non-finite scalar output.")
    return value


def gradcheck(
    function: Callable[[Tensor], Tensor],
    input: Tensor | object,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> GradientCheckResult:
    """Compare autodiff gradients with central finite differences.

    ``function`` must map one Tensor argument to a scalar Tensor. The input is
    never mutated. A fresh Tensor and computation graph are created for every
    analytical or numerical evaluation.

    The numerical estimate for element i is
    ``(f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)``.
    """
    if not callable(function):
        raise TypeError("function must be callable.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number.")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number.")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be a finite non-negative number.")

    original = input.data.copy() if isinstance(input, Tensor) else np.asarray(input, dtype=np.float64).copy()
    if not np.issubdtype(original.dtype, np.number):
        raise TypeError("input must contain numeric values.")
    if not np.all(np.isfinite(original)):
        raise ValueError("input must contain only finite values.")
    original = original.astype(np.float64, copy=True)

    analytical_input = Tensor(original, requires_grad=True)
    analytical_output = function(analytical_input)
    if not isinstance(analytical_output, Tensor):
        raise TypeError("The function passed to gradcheck must return a Tensor.")
    if analytical_output.data.size != 1:
        raise ValueError(
            "gradcheck requires a scalar Tensor output. Reduce the output, "
            "for example with .sum() or .mean()."
        )
    if not np.all(np.isfinite(analytical_output.data)):
        raise ValueError("The function produced a non-finite scalar output.")
    analytical_output.backward()
    if analytical_input.grad is None:
        raise RuntimeError("Autodiff did not produce an input gradient.")
    analytical = analytical_input.grad.copy()

    numerical = np.zeros_like(original, dtype=np.float64)
    for index in np.ndindex(original.shape):
        plus = original.copy()
        minus = original.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical[index] = (
            _scalar_value(function, plus) - _scalar_value(function, minus)
        ) / (2.0 * eps)

    difference = np.abs(analytical - numerical)
    denominator = np.maximum(np.maximum(np.abs(analytical), np.abs(numerical)), np.finfo(np.float64).tiny)
    max_absolute_error = float(np.max(difference)) if difference.size else 0.0
    max_relative_error = float(np.max(difference / denominator)) if difference.size else 0.0
    passed = bool(np.allclose(analytical, numerical, atol=atol, rtol=rtol))

    return GradientCheckResult(
        analytical=analytical,
        numerical=numerical,
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        passed=passed,
        eps=float(eps),
        atol=float(atol),
        rtol=float(rtol),
    )
