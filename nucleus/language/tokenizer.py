"""Deterministic character-level tokenizer."""
class CharTokenizer:
    def __init__(self, text: str):
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string.")
        self.vocab = tuple(sorted(set(text)))
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
    @property
    def vocab_size(self):
        return len(self.vocab)
    def encode(self, text: str):
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"Unknown character: {exc.args[0]!r}") from exc
    def decode(self, token_ids):
        result = []
        for token in token_ids:
            i = int(token)
            if i not in self.itos:
                raise ValueError(f"Unknown token ID: {i}")
            result.append(self.itos[i])
        return "".join(result)
