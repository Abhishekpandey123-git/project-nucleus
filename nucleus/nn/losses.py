"""Loss functions for supervised learning."""
from nucleus.core.tensor import Tensor


def mean_squared_error(predictions: Tensor, targets: Tensor) -> Tensor:
    """Return the mean squared error between predictions and targets."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have the same shape; got "
            f"{predictions.shape} and {targets.shape}."
        )
    return ((predictions - targets) ** 2).mean()
