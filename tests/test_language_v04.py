import numpy as np
import pytest
from nucleus import Tensor
from nucleus.language import CharTokenizer, CharacterDataset, CharacterLanguageModel, batch_iterator, generate
from nucleus.nn.embedding import Embedding
from nucleus.nn.losses import cross_entropy
from nucleus.optim import SGD

def test_tokenizer_roundtrip():
    tok = CharTokenizer("banana!")
    assert tok.decode(tok.encode("banana!")) == "banana!"
    with pytest.raises(ValueError): tok.encode("z")

def test_dataset_windows_and_batches():
    ds = CharacterDataset([1,2,3,4,5], 2)
    np.testing.assert_array_equal(ds[0][0], [1,2])
    np.testing.assert_array_equal(ds[0][1], [2,3])
    assert sum(len(x) for x,_ in batch_iterator(ds, 2)) == len(ds)

def test_embedding_backprop():
    layer = Embedding(5, 3, seed=2)
    out = layer(np.array([[1,2],[3,1]]))
    sum(x.sum() for x in out).backward()
    assert np.any(layer.weight.grad != 0)

def test_cross_entropy_backprop():
    logits = Tensor([[2.,0.,-1.]], requires_grad=True)
    loss = cross_entropy(logits, [0]); loss.backward()
    assert logits.grad[0,0] < 0 and logits.grad[0,1] > 0

def test_rnn_shapes_parameters_and_update():
    model = CharacterLanguageModel(5, 4, 6, seed=1)
    x = np.array([[1,2,3],[3,2,1]])
    y = np.array([[2,3,4],[2,1,0]])
    logits = model(x)
    assert len(logits) == 3 and all(z.shape == (2,5) for z in logits)
    opt = SGD(model.parameters(), learning_rate=.05)
    loss = sum(cross_entropy(z,y[:,t]) for t,z in enumerate(logits))/len(logits)
    opt.zero_grad(); loss.backward()
    before = [p.data.copy() for p in model.parameters()]
    opt.step()
    assert any(not np.array_equal(a,p.data) for a,p in zip(before,model.parameters()))

def test_generation_seeded():
    tok = CharTokenizer("abcabcabc")
    model = CharacterLanguageModel(tok.vocab_size, 4, 6, seed=1)
    assert generate(model,tok,"a",8,seed=9) == generate(model,tok,"a",8,seed=9)
