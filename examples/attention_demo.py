import numpy as np
from nucleus import Tensor
from nucleus.nn.transformer import TransformerBlock

rng = np.random.default_rng(7)
block = TransformerBlock(16, 4, 32, seed=7)
tokens = Tensor(rng.normal(size=(2, 6, 16)), requires_grad=True)
outputs = block(tokens, causal=True)
loss = (outputs ** 2).mean()
loss.backward()
print("output shape:", outputs.shape)
print("loss:", float(loss.data))
print("parameter count:", len(block.parameters()))
print("finite input gradients:", bool(np.isfinite(tokens.grad).all()))
