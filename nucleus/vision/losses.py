"""Differentiable classification losses for vision experiments."""
import numpy as np
from nucleus.core.tensor import Tensor, is_grad_enabled


def cross_entropy(logits: Tensor, targets):
    """Mean softmax cross-entropy for logits (N,C) and integer labels (N,)."""
    y=np.asarray(targets,dtype=np.int64)
    if logits.ndim!=2 or y.shape!=(logits.shape[0],): raise ValueError("expected logits (N,C) and targets (N,).")
    n,c=logits.shape
    if np.any(y<0) or np.any(y>=c): raise ValueError("target class index is out of range.")
    shifted=logits.data-logits.data.max(axis=1,keepdims=True)
    exp=np.exp(shifted); probs=exp/exp.sum(axis=1,keepdims=True)
    loss=float(-np.log(np.maximum(probs[np.arange(n),y],np.finfo(float).tiny)).mean())
    out=Tensor(loss,is_grad_enabled() and logits.requires_grad)
    if logits.requires_grad:
        out._parents=(logits,)
        def backward():
            if out.grad is None:return
            grad=probs.copy();grad[np.arange(n),y]-=1.0
            assert logits.grad is not None
            logits.grad += grad*(float(out.grad)/n)
        out._backward=backward
    return out
