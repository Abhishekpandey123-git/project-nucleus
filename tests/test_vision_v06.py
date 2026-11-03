import numpy as np
import pytest
from nucleus import Tensor
from nucleus.vision import Conv2D, MaxPool2D, AvgPool2D, Flatten, ImageDataset, cross_entropy, SmallCNN


def finite_difference(fn,x,eps=1e-6):
    result=np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        plus=x.copy();minus=x.copy();plus[idx]+=eps;minus[idx]-=eps
        result[idx]=(fn(plus)-fn(minus))/(2*eps)
    return result


def test_conv2d_shape_and_backward():
    rng=np.random.default_rng(1)
    x=Tensor(rng.normal(size=(2,1,5,5)),True)
    layer=Conv2D(1,2,3,seed=2)
    y=layer(x)
    assert y.shape==(2,2,3,3)
    (y**2).mean().backward()
    assert np.isfinite(x.grad).all() and np.isfinite(layer.weight.grad).all()


def test_conv2d_input_gradient_matches_finite_difference():
    rng=np.random.default_rng(4);data=rng.normal(size=(1,1,4,4))
    layer=Conv2D(1,1,2,seed=3)
    x=Tensor(data,True);layer(x).sum().backward()
    weights=layer.weight.data.copy();bias=layer.bias.data.copy()
    def objective(z):
        total=0.0
        for i in range(3):
            for j in range(3):
                total += np.sum(z[0,0,i:i+2,j:j+2]*weights[0,0])+bias[0]
        return total
    numeric=finite_difference(objective,data)
    np.testing.assert_allclose(x.grad,numeric,atol=1e-6)


def test_max_pool_gradient_routes_to_maximum():
    x=Tensor(np.array([[[[1.,4.],[3.,2.]]]]),True)
    MaxPool2D(2)(x).sum().backward()
    np.testing.assert_array_equal(x.grad,[[[[0.,1.],[0.,0.]]]])


def test_average_pool_gradient():
    x=Tensor(np.ones((1,1,2,2)),True)
    AvgPool2D(2)(x).sum().backward()
    np.testing.assert_allclose(x.grad,np.full((1,1,2,2),.25))


def test_flatten_and_image_dataset():
    x=Tensor(np.ones((3,2,4,4)),True);y=Flatten()(x)
    assert y.shape==(3,32);y.sum().backward();np.testing.assert_allclose(x.grad,1.)
    ds=ImageDataset(np.full((5,8,8),255),[0,1,0,1,0])
    assert ds.images.shape==(5,1,8,8) and ds.images.max()==1.
    assert sum(len(batch_y) for _,batch_y in ds.batches(2))==5


def test_cross_entropy_gradient():
    logits=Tensor([[2.,1.,0.],[0.,1.,2.]],True)
    loss=cross_entropy(logits,[0,2]);loss.backward()
    assert loss.data>0 and np.isfinite(logits.grad).all()
    np.testing.assert_allclose(logits.grad.sum(axis=1),0.,atol=1e-12)


def test_small_cnn_forward_backward_and_parameter_discovery():
    model=SmallCNN(in_channels=1,num_classes=3)
    x=Tensor(np.random.default_rng(0).normal(size=(2,1,8,8)),True)
    logits=model(x)
    assert logits.shape==(2,3)
    loss=cross_entropy(logits,[0,2]);loss.backward()
    assert len(model.parameters())>0
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_shapes():
    with pytest.raises(ValueError): Conv2D(1,2,3)(Tensor(np.ones((1,5,5))))
    with pytest.raises(ValueError): ImageDataset(np.ones((2,4,4)),[0])
