"""SGD optimizer with optional momentum and serializable state."""
from __future__ import annotations

import numpy as np
from nucleus.core.tensor import Tensor


class SGD:
    """Stochastic gradient descent.

    ``momentum=0`` is ordinary SGD. With momentum, the velocity is
    ``v = momentum * v + grad`` and parameters update by ``lr * v``.
    """

    def __init__(self, parameters: list[Tensor], learning_rate: float = 0.01, momentum: float = 0.0):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("momentum must be in the interval [0, 1).")
        self.parameters = list(parameters)
        if any(not isinstance(parameter, Tensor) for parameter in self.parameters):
            raise TypeError("optimizer parameters must be Tensor instances.")
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self._velocity: dict[int, np.ndarray] = {}

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            if self.momentum:
                velocity = self._velocity.get(index)
                if velocity is None:
                    velocity = np.zeros_like(parameter.data)
                velocity = self.momentum * velocity + parameter.grad
                self._velocity[index] = velocity.copy()
                update = velocity
            else:
                update = parameter.grad
            parameter.data -= self.learning_rate * update

    def state_dict(self) -> dict[str, object]:
        """Return a deep-copy snapshot suitable for checkpointing."""
        return {
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "velocity": {index: value.copy() for index, value in self._velocity.items()},
            "parameter_shapes": [parameter.shape for parameter in self.parameters],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore optimizer hyperparameters and momentum buffers safely."""
        required = {"learning_rate", "momentum", "velocity", "parameter_shapes"}
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("Invalid SGD state_dict: missing required keys.")
        lr = float(state["learning_rate"])
        momentum = float(state["momentum"])
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("state learning_rate must be finite and positive.")
        if not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError("state momentum must be in the interval [0, 1).")
        shapes = list(state["parameter_shapes"])
        if shapes != [parameter.shape for parameter in self.parameters]:
            raise ValueError("Optimizer state parameter shapes do not match this optimizer.")
        raw_velocity = state["velocity"]
        if not isinstance(raw_velocity, dict):
            raise ValueError("state velocity must be a dictionary.")
        velocity: dict[int, np.ndarray] = {}
        for raw_index, raw_value in raw_velocity.items():
            index = int(raw_index)
            if index < 0 or index >= len(self.parameters):
                raise ValueError("Optimizer state contains an invalid parameter index.")
            value = np.asarray(raw_value, dtype=np.float64)
            if value.shape != self.parameters[index].shape or not np.all(np.isfinite(value)):
                raise ValueError("Optimizer state contains an invalid momentum buffer.")
            velocity[index] = value.copy()
        self.learning_rate = lr
        self.momentum = momentum
        self._velocity = velocity
