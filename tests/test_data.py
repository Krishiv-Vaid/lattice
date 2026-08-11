import random

import pytest

from lattice.data import (
    DataLoader,
    TensorDataset,
)
from lattice.tensor import Tensor


def test_tensor_dataset_length():
    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    y = Tensor([
        0.0,
        1.0,
        2.0,
    ])

    dataset = TensorDataset(
        x,
        y,
    )

    assert len(dataset) == 3


def test_tensor_dataset_rejects_mismatched_lengths():
    x = Tensor([
        [1.0],
        [2.0],
    ])

    y = Tensor([
        0.0,
        1.0,
        2.0,
    ])

    with pytest.raises(ValueError):
        TensorDataset(
            x,
            y,
        )


def test_dataloader_single_batch():
    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    y = Tensor([
        0.0,
        1.0,
    ])

    dataset = TensorDataset(
        x,
        y,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
    )

    batches = list(loader)

    assert len(batches) == 1

    batch_x, batch_y = batches[0]

    assert batch_x.shape == (2, 2)
    assert batch_y.shape == (2,)

    assert batch_x.data == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    assert batch_y.data == [
        0.0,
        1.0,
    ]


def test_dataloader_multiple_batches():
    x = Tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])

    y = Tensor([
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ])

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    batches = list(loader)

    assert len(batches) == 3

    assert batches[0][0].data == [
        1.0,
        2.0,
    ]

    assert batches[1][0].data == [
        3.0,
        4.0,
    ]

    assert batches[2][0].data == [
        5.0,
    ]


def test_dataloader_length():
    x = Tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])

    y = Tensor([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    assert len(loader) == 3


def test_dataloader_shuffle():
    random.seed(42)

    x = Tensor([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = Tensor([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=5,
        shuffle=True,
    )

    batch_x, batch_y = next(
        iter(loader)
    )

    assert batch_x.data != [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    assert batch_x.data == batch_y.data


def test_dataloader_supports_higher_dimensions():
    x = Tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
    ])

    y = Tensor([
        0.0,
        1.0,
    ])

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=2,
    )

    batch_x, batch_y = next(
        iter(loader)
    )

    assert batch_x.shape == (
        2,
        2,
        2,
    )

    assert batch_x.data == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ]

    assert batch_y.data == [
        0.0,
        1.0,
    ]


def test_dataloader_rejects_invalid_batch_size():
    dataset = TensorDataset(
        Tensor([
            [1.0],
        ])
    )

    with pytest.raises(ValueError):
        DataLoader(
            dataset,
            batch_size=0,
        )