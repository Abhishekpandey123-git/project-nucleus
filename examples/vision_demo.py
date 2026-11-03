"""Synthetic CPU-only CNN training smoke test; no external dataset required."""
import numpy as np
from nucleus.core.tensor import Tensor
from nucleus.optim import SGD
from nucleus.vision import ImageDataset, SmallCNN, cross_entropy


def main():
    rng=np.random.default_rng(12)
    images=np.zeros((60,8,8),dtype=np.float64);labels=np.arange(60)%3
    # Three simple visual classes: vertical, horizontal, and diagonal strokes.
    for i,label in enumerate(labels):
        if label==0: images[i,:,3:5]=1.
        elif label==1: images[i,3:5,:]=1.
        else: np.fill_diagonal(images[i],1.)
        images[i]+=rng.normal(0,.04,(8,8));images[i]=np.clip(images[i],0,1)
    dataset=ImageDataset(images,labels,normalize=False)
    model=SmallCNN(1,3,seed=4);optimizer=SGD(model.parameters(),learning_rate=.025)
    for epoch in range(12):
        losses=[]
        for xb,yb in dataset.batches(10,shuffle=True,seed=epoch):
            logits=model(Tensor(xb,requires_grad=True));loss=cross_entropy(logits,yb)
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.data))
        print(f"epoch {epoch+1:02d} loss {np.mean(losses):.4f}")
    logits=model(Tensor(dataset.images));pred=logits.data.argmax(axis=1)
    print(f"training accuracy: {(pred==labels).mean():.1%}")

if __name__=="__main__":main()
