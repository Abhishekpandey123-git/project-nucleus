"""Small, dependency-light image dataset and batching utilities."""
from __future__ import annotations
import numpy as np


def normalize_images(images, scale=True):
    """Convert image arrays to float64 NCHW; scale byte-like data to [0, 1]."""
    x=np.asarray(images,dtype=np.float64)
    if x.ndim==3: x=x[:,None,:,:]  # N,H,W -> N,1,H,W
    if x.ndim!=4: raise ValueError("images must have shape (N,H,W) or (N,C,H,W).")
    if not np.all(np.isfinite(x)): raise ValueError("images must contain only finite values.")
    if scale and x.size and x.max()>1.0:
        if x.min()<0 or x.max()>255: raise ValueError("expected image values in [0,1] or [0,255].")
        x=x/255.0
    return x


class ImageDataset:
    """In-memory image/label dataset. Images are stored as normalized NCHW arrays."""
    def __init__(self, images, labels, normalize=True):
        self.images=normalize_images(images,scale=normalize)
        self.labels=np.asarray(labels,dtype=np.int64)
        if self.labels.ndim!=1 or len(self.labels)!=len(self.images):
            raise ValueError("labels must be a 1D array with one label per image.")
        if len(self.images)==0: raise ValueError("dataset cannot be empty.")

    def __len__(self): return len(self.labels)
    def __getitem__(self,index): return self.images[index],int(self.labels[index])

    def batches(self,batch_size,shuffle=False,seed=None):
        if not isinstance(batch_size,int) or batch_size<=0: raise ValueError("batch_size must be a positive integer.")
        indices=np.arange(len(self))
        if shuffle: np.random.default_rng(seed).shuffle(indices)
        for start in range(0,len(self),batch_size):
            selected=indices[start:start+batch_size]
            yield self.images[selected],self.labels[selected]
