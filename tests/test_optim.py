from lattice.optim import SGD
from lattice.tensor import Tensor
from lattice.value import Value


def test_sgd_step():
    parameter = Value(2.0)
    parameter.grad = 3.0

    optimizer = SGD(
        [parameter],
        lr=0.1,
    )

    optimizer.step()

    assert parameter.data == 1.7


def test_sgd_zero_grad():
    parameter = Value(2.0)
    parameter.grad = 3.0

    optimizer = SGD(
        [parameter],
        lr=0.1,
    )

    optimizer.zero_grad()

    assert parameter.grad == 0.0


def test_tensor_sgd_step():
    parameter = Tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        requires_grad=True,
    )

    parameter.grad = [
        10.0,
        20.0,
        30.0,
    ]

    optimizer = SGD(
        [parameter],
        lr=0.1,
    )

    optimizer.step()

    assert parameter.data == [
        0.0,
        0.0,
        0.0,
    ]


def test_tensor_sgd_zero_grad():
    parameter = Tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        requires_grad=True,
    )

    parameter.grad = [
        10.0,
        20.0,
        30.0,
    ]

    optimizer = SGD(
        [parameter],
        lr=0.1,
    )

    optimizer.zero_grad()

    assert parameter.grad == [
        0.0,
        0.0,
        0.0,
    ]


def test_tensor_sgd_multiple_parameters():
    weight = Tensor(
        [
            [1.0],
            [2.0],
        ],
        requires_grad=True,
    )

    bias = Tensor(
        [3.0],
        requires_grad=True,
    )

    weight.grad = [
        4.0,
        5.0,
    ]

    bias.grad = [
        6.0,
    ]

    optimizer = SGD(
        [weight, bias],
        lr=0.1,
    )

    optimizer.step()

    assert weight.data == [
        0.6,
        1.5,
    ]

    assert bias.data == [
        2.4,
    ]


def test_tensor_sgd_ignores_non_trainable_tensor():
    parameter = Tensor([
        1.0,
        2.0,
    ])

    optimizer = SGD(
        [parameter],
        lr=0.1,
    )

    optimizer.step()

    assert parameter.data == [
        1.0,
        2.0,
    ]


def test_sgd_rejects_nonpositive_learning_rate():
    parameter = Value(1.0)

    try:
        SGD(
            [parameter],
            lr=0.0,
        )

        assert False

    except ValueError:
        pass