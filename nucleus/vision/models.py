"""Small convolutional image classifier assembled from Nucleus modules."""
from nucleus.core.tensor import Tensor
from nucleus.nn.layers import Linear, Module, ReLU
from nucleus.vision.layers import Conv2D, MaxPool2D, Flatten


class SmallCNN(Module):
    """Two-convolution classifier for small NCHW images (e.g. 1x8x8)."""
    def __init__(self,in_channels=1,num_classes=10,seed=7):
        self.conv1=Conv2D(in_channels,8,3,seed=seed)
        self.conv2=Conv2D(8,16,3,seed=seed+1)
        self.pool=MaxPool2D(2)
        self.flatten=Flatten()
        # 8x8 -> conv 6x6 -> pool 3x3 -> conv 1x1 -> 16 features
        self.classifier=Linear(16,num_classes,seed=seed+2)

    def __call__(self,x:Tensor)->Tensor:
        x=self.pool(self.conv1(x).relu())
        x=self.conv2(x).relu()
        return self.classifier(self.flatten(x))
