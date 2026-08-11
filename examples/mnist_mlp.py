import random

from torchvision.datasets import MNIST

import lattice.nn as nn
import lattice.optim as optim
from lattice.data import DataLoader, TensorDataset
from lattice.tensor import Tensor


TRAIN_SAMPLES = 256
TEST_SAMPLES = 128
BATCH_SIZE = 16
HIDDEN_SIZE = 32
EPOCHS = 5


def image_to_vector(image):
    return [
        float(pixel) / 255.0
        for pixel in image.getdata()
    ]


def load_subset(dataset, count):
    features = []
    targets = []

    for index in range(count):
        image, label = dataset[index]

        features.append(
            image_to_vector(image)
        )

        targets.append(
            float(label)
        )

    return (
        Tensor(features),
        Tensor(targets),
    )


def calculate_accuracy(
    model,
    data_loader,
):
    correct = 0
    total = 0

    for batch_x, batch_y in data_loader:
        logits = model(batch_x)

        predictions = logits.argmax(
            dim=1
        )

        for index in range(
            batch_y.shape[0]
        ):
            predicted = int(
                predictions[index]
            )

            target = int(
                batch_y[index]
            )

            if predicted == target:
                correct += 1

            total += 1

    return correct / total


def main():
    random.seed(42)

    train_dataset = MNIST(
        root="data",
        train=True,
        download=True,
    )

    test_dataset = MNIST(
        root="data",
        train=False,
        download=True,
    )

    print("Loading MNIST subset...")

    train_x, train_y = load_subset(
        train_dataset,
        TRAIN_SAMPLES,
    )

    test_x, test_y = load_subset(
        test_dataset,
        TEST_SAMPLES,
    )

    train_loader = DataLoader(
        TensorDataset(
            train_x,
            train_y,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        TensorDataset(
            test_x,
            test_y,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Linear(
            28 * 28,
            HIDDEN_SIZE,
        ),
        nn.ReLU(),
        nn.Linear(
            HIDDEN_SIZE,
            10,
        ),
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    print()
    print("Training Lattice MNIST MLP...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        batch_count = 0

        for batch_x, batch_y in train_loader:
            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            batch_count += 1

        average_loss = (
            total_loss / batch_count
        )

        train_accuracy = calculate_accuracy(
            model,
            train_loader,
        )

        test_accuracy = calculate_accuracy(
            model,
            test_loader,
        )

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} "
            f"loss={average_loss:.4f} "
            f"train_acc="
            f"{train_accuracy * 100:.1f}% "
            f"test_acc="
            f"{test_accuracy * 100:.1f}%"
        )

    print()
    print("Final test predictions:")

    batch_x, batch_y = next(
        iter(test_loader)
    )

    logits = model(batch_x)

    predictions = logits.argmax(
        dim=1
    )

    for index in range(
        min(10, batch_y.shape[0])
    ):
        print(
            f"sample {index:2d}: "
            f"predicted="
            f"{int(predictions[index])} "
            f"target="
            f"{int(batch_y[index])}"
        )


if __name__ == "__main__":
    main()