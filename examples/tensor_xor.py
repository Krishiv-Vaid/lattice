import random

import lattice as lt
import lattice.nn as nn
import lattice.optim as optim


def main():
    random.seed(42)

    x = lt.Tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    y = lt.Tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ])

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    model[0].bias.data = [
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    criterion = nn.TensorMSELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.03,
    )

    for epoch in range(10000):
        prediction = model(x)

        loss = criterion(
            prediction,
            y,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch:5d} "
                f"Loss {loss.data[0]:.6f}"
            )

    prediction = model(x)

    print()
    print("Final predictions:")

    labels = [
        "[0, 0]",
        "[0, 1]",
        "[1, 0]",
        "[1, 1]",
    ]

    for label, value in zip(
        labels,
        prediction.data,
    ):
        print(
            f"{label} -> {value:.6f}"
        )


if __name__ == "__main__":
    main()