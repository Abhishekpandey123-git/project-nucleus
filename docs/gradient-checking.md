# Numerical gradient checking

Project Nucleus v0.2 adds `nucleus.testing.gradcheck`, which compares reverse-mode autodiff with central finite differences.

```python
from nucleus import Tensor
from nucleus.testing import gradcheck

x = Tensor([0.2, -0.7, 1.4])
result = gradcheck(lambda value: (value.tanh() ** 2).sum(), x)

print(result.passed)
print(result.max_absolute_error)
print(result.analytical)
print(result.numerical)
```

The checked function must return a scalar `Tensor`; use `.sum()` or `.mean()` for vector/matrix outputs. The input is not mutated. Finite differences are approximate, so tolerances and `eps` are configurable.

The checker is intended for small tensors and tests, not training: it evaluates the function twice per input element in addition to one analytical pass, making it expensive for large tensors.
