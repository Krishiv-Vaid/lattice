import random

import lattice as lt
import lattice.nn as nn
import lattice.optim as optim


def main():
    random.seed(42)

    x = lt.Tensor([
        [-2.0, -1.0],
        [-1.5, -0.5],
        [-1.0, -1.5],

        [1.0, -1.5],
        [1.5, -0.5],
        [2.0, -1.0],

        [-0.5, 1.0],
        [0.0, 2.0],
        [0.5, 1.5],
    ])

    targets = lt.Tensor([
        0.0,
        0.0,
        0.0,

        1.0,
        1.0,
        1.0,

        2.0,
        2.0,
        2.0,
    ])

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )

    model[0].bias.data = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.03,
    )

    for epoch in range(2000):
        logits = model(x)

        loss = criterion(
            logits,
            targets,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(
                f"Epoch {epoch:4d} "
                f"Loss {loss.data[0]:.6f}"
            )

    logits = model(x)
    probabilities = logits.softmax(dim=1)

    print()
    print("Final predictions:")

    for sample_index in range(
        x.shape[0]
    ):
        row_start = (
            sample_index
            * probabilities.shape[1]
        )

        row = probabilities.data[
            row_start:
            row_start
            + probabilities.shape[1]
        ]

        predicted_class = max(
            range(len(row)),
            key=lambda index: row[index],
        )

        target_class = int(
            targets[sample_index]
        )

        print(
            f"sample {sample_index}: "
            f"target={target_class} "
            f"predicted={predicted_class} "
            f"probs={row}"
        )


if __name__ == "__main__":
    main()