import lattice as lt
import lattice.nn as nn
import lattice.optim as optim


def main():
    x = lt.Tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = lt.Tensor([
        [2.0],
        [4.0],
        [6.0],
        [8.0],
    ])

    model = nn.Linear(
        in_features=1,
        out_features=1,
    )

    criterion = nn.TensorMSELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
    )

    for epoch in range(1000):
        prediction = model(x)

        loss = criterion(
            prediction,
            y,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} "
                f"Loss {loss.data[0]:.6f}"
            )

    print()

    print(
        "Learned weight:",
        model.weight.data,
    )

    print(
        "Learned bias:",
        model.bias.data,
    )

    test_input = lt.Tensor([
        [5.0],
    ])

    test_output = model(
        test_input
    )

    print(
        "Prediction for x=5:",
        test_output.data,
    )


if __name__ == "__main__":
    main()