"""Train a tiny character-level RNN on a toy corpus."""
import numpy as np
from nucleus.language import CharTokenizer, CharacterDataset, CharacterLanguageModel, batch_iterator, generate
from nucleus.nn.losses import cross_entropy
from nucleus.optim import SGD

TEXT = ("project nucleus learns one character at a time. small models help us study language. " * 30)
def main():
    tokenizer = CharTokenizer(TEXT)
    dataset = CharacterDataset(tokenizer.encode(TEXT), context_length=24)
    model = CharacterLanguageModel(tokenizer.vocab_size, 16, 32, seed=7)
    optimizer = SGD(model.parameters(), learning_rate=0.08)
    for epoch in range(12):
        losses = []
        for x, y in batch_iterator(dataset, 16, shuffle=True, seed=epoch):
            logits = model(x)
            loss = sum(cross_entropy(step, y[:, t]) for t, step in enumerate(logits)) / len(logits)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.data))
        print(f"epoch={epoch+1:02d} loss={np.mean(losses):.4f}")
    print(generate(model, tokenizer, "project ", 120, seed=3))
if __name__ == "__main__": main()
