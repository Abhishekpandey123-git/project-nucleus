"""Autoregressive character sampling."""
import numpy as np
from nucleus.core.tensor import no_grad

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, seed=None):
    if not prompt: raise ValueError("prompt must not be empty.")
    if max_new_tokens < 0: raise ValueError("max_new_tokens must be non-negative.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    rng = np.random.default_rng(seed)
    ids = tokenizer.encode(prompt)
    with no_grad():
        for _ in range(max_new_tokens):
            context = np.asarray([ids[-128:]], dtype=np.int64)
            logits = model(context)[-1].data[0] / temperature
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            ids.append(int(rng.choice(len(probs), p=probs)))
    return tokenizer.decode(ids)
