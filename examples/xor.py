"""Train a tiny MLP on XOR using the Project Nucleus v0.1 engine."""
from nucleus import Tensor
from nucleus.nn import Linear, Sequential, Tanh
from nucleus.nn.losses import mean_squared_error
from nucleus.optim import SGD


def main() -> None:
    inputs = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = Tensor([[0], [1], [1], [0]])

    model = Sequential(Linear(2, 4, seed=11), Tanh(), Linear(4, 1, seed=12))
    optimizer = SGD(model.parameters(), learning_rate=0.08)

    for step in range(1201):
        predictions = model(inputs)
        loss = mean_squared_error(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"step={step:4d} loss={float(loss.data):.6f}")

    print("Predictions:")
    print(model(inputs).data)


if __name__ == "__main__":
    main()
