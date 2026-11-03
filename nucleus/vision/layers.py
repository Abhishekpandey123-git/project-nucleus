"""NumPy-first differentiable vision layers for Project Nucleus."""
from __future__ import annotations
import numpy as np
from nucleus.core.tensor import Tensor, is_grad_enabled
from nucleus.nn.layers import Module
from nucleus.nn.parameter import Parameter


def _pair(value, name):
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an int or a pair of ints.")
    if min(result) <= 0:
        raise ValueError(f"{name} values must be positive.")
    return result


class Conv2D(Module):
    """Valid 2D convolution. Input/output layout is NCHW; no implicit padding."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, seed=None):
        if min(in_channels, out_channels) <= 0:
            raise ValueError("channel counts must be positive.")
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size, self.stride = _pair(kernel_size, "kernel_size"), _pair(stride, "stride")
        kh, kw = self.kernel_size
        rng = np.random.default_rng(seed)
        fan_in = in_channels * kh * kw
        fan_out = out_channels * kh * kw
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.weight = Parameter(rng.uniform(-limit, limit, (out_channels, in_channels, kh, kw)))
        self.bias = Parameter(np.zeros((out_channels,)))

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"Conv2D expects NCHW input with {self.in_channels} channels.")
        x = inputs.data
        n, _, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if h < kh or w < kw:
            raise ValueError("kernel cannot be larger than the input spatial dimensions.")
        oh, ow = (h-kh)//sh+1, (w-kw)//sw+1
        windows = np.empty((n, self.in_channels, oh, ow, kh, kw), dtype=x.dtype)
        for iy in range(kh):
            for ix in range(kw):
                windows[..., iy, ix] = x[:, :, iy:iy+oh*sh:sh, ix:ix+ow*sw:sw]
        out_data = np.einsum("ncyxij,ocij->noyx", windows, self.weight.data) + self.bias.data[None, :, None, None]
        requires = is_grad_enabled() and (inputs.requires_grad or self.weight.requires_grad or self.bias.requires_grad)
        out = Tensor(out_data, requires)
        if requires:
            out._parents = tuple(t for t in (inputs, self.weight, self.bias) if t.requires_grad)
            def backward():
                if out.grad is None: return
                g = out.grad
                if self.weight.requires_grad:
                    assert self.weight.grad is not None
                    self.weight.grad += np.einsum("noyx,ncyxij->ocij", g, windows)
                if self.bias.requires_grad:
                    assert self.bias.grad is not None
                    self.bias.grad += g.sum(axis=(0, 2, 3))
                if inputs.requires_grad:
                    assert inputs.grad is not None
                    dx = np.zeros_like(x)
                    for iy in range(kh):
                        for ix in range(kw):
                            contribution = np.einsum("noyx,oc->ncyx", g, self.weight.data[:, :, iy, ix])
                            dx[:, :, iy:iy+oh*sh:sh, ix:ix+ow*sw:sw] += contribution
                    inputs.grad += dx
            out._backward = backward
        return out


class MaxPool2D(Module):
    """Non-overlapping/strided max pooling with first-maximum gradient routing."""
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride if stride is not None else kernel_size, "stride")

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4: raise ValueError("MaxPool2D expects NCHW input.")
        n,c,h,w = inputs.shape; kh,kw=self.kernel_size; sh,sw=self.stride
        if h < kh or w < kw: raise ValueError("pool window cannot exceed input dimensions.")
        oh,ow=(h-kh)//sh+1,(w-kw)//sw+1
        windows=np.empty((n,c,oh,ow,kh,kw),dtype=inputs.data.dtype)
        for iy in range(kh):
            for ix in range(kw): windows[...,iy,ix]=inputs.data[:,:,iy:iy+oh*sh:sh,ix:ix+ow*sw:sw]
        flat=windows.reshape(n,c,oh,ow,kh*kw); indices=flat.argmax(axis=-1)
        out_data=flat.max(axis=-1); out=Tensor(out_data,is_grad_enabled() and inputs.requires_grad)
        if inputs.requires_grad:
            out._parents=(inputs,)
            def backward():
                if out.grad is None:return
                dx=np.zeros_like(inputs.data)
                for iy in range(kh):
                    for ix in range(kw):
                        mask=(indices==iy*kw+ix)
                        dx[:,:,iy:iy+oh*sh:sh,ix:ix+ow*sw:sw] += out.grad*mask
                assert inputs.grad is not None; inputs.grad += dx
            out._backward=backward
        return out


class AvgPool2D(Module):
    """Average pooling over valid windows."""
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size=_pair(kernel_size,"kernel_size")
        self.stride=_pair(stride if stride is not None else kernel_size,"stride")

    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim!=4: raise ValueError("AvgPool2D expects NCHW input.")
        n,c,h,w=inputs.shape;kh,kw=self.kernel_size;sh,sw=self.stride
        if h<kh or w<kw: raise ValueError("pool window cannot exceed input dimensions.")
        oh,ow=(h-kh)//sh+1,(w-kw)//sw+1
        windows=np.empty((n,c,oh,ow,kh,kw),dtype=inputs.data.dtype)
        for iy in range(kh):
            for ix in range(kw):windows[...,iy,ix]=inputs.data[:,:,iy:iy+oh*sh:sh,ix:ix+ow*sw:sw]
        out=Tensor(windows.mean(axis=(-1,-2)),is_grad_enabled() and inputs.requires_grad)
        if inputs.requires_grad:
            out._parents=(inputs,)
            def backward():
                if out.grad is None:return
                dx=np.zeros_like(inputs.data); contribution=out.grad/(kh*kw)
                for iy in range(kh):
                    for ix in range(kw):dx[:,:,iy:iy+oh*sh:sh,ix:ix+ow*sw:sw]+=contribution
                assert inputs.grad is not None;inputs.grad+=dx
            out._backward=backward
        return out


class Flatten(Module):
    """Flatten all dimensions after the batch dimension."""
    def __call__(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 2: raise ValueError("Flatten expects a batch dimension and at least one feature dimension.")
        return inputs.reshape(inputs.shape[0], -1)
