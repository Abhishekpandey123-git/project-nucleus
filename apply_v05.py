from pathlib import Path

p = Path("nucleus/core/tensor.py")
s = p.read_text()
a = s.index("    def __matmul__(self, other: object) -> Tensor:\n")
b = s.index("    def sum(", a)
methods = """
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

"""
p.write_text(s[:a] + methods + s[b:])

p = Path("nucleus/nn/layers.py")
lines = p.read_text().splitlines()
for i, line in enumerate(lines):
    if "if inputs.ndim != 2 or inputs.shape[1] != self.weight.shape[0]:" in line:
        lines[i:i+2] = [
            "        if inputs.ndim < 2 or inputs.shape[-1] != self.weight.shape[0]:",
            "            raise ValueError(",
            '                f"Expected input with final dimension {self.weight.shape[0]}, got {inputs.shape}."',
            "            )",
        ]
        break
else:
    raise SystemExit("Linear validation line not found.")
p.write_text("\\n".join(lines) + "\\n")
print("Patched Tensor and Linear for v0.5.")
