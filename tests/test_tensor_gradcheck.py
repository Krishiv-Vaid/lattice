import pytest

from lattice.gradcheck import (
    gradcheck,
    numerical_gradient,
)
from lattice.nn import CrossEntropyLoss
from lattice.tensor import Tensor


def test_numerical_gradient_square():
    x = Tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        requires_grad=True,
    )

    def function():
        return (
            x * x
        ).sum()

    gradient = numerical_gradient(
        function,
        x,
    )

    assert gradient == pytest.approx([
        2.0,
        4.0,
        6.0,
    ])


def test_gradcheck_elementwise_multiply():
    x = Tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        requires_grad=True,
    )

    y = Tensor(
        [
            4.0,
            5.0,
            6.0,
        ],
        requires_grad=True,
    )

    def function():
        return (
            x * y
        ).sum()

    assert gradcheck(
        function,
        [x, y],
    )


def test_gradcheck_broadcast_addition():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    bias = Tensor(
        [0.5, -1.0, 2.0],
        requires_grad=True,
    )

    def function():
        output = x + bias

        return (
            output * output
        ).mean()

    assert gradcheck(
        function,
        [x, bias],
    )


def test_gradcheck_matmul():
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [0.5, -1.0],
            [2.0, 0.25],
        ],
        requires_grad=True,
    )

    def function():
        output = a @ b

        return (
            output * output
        ).mean()

    assert gradcheck(
        function,
        [a, b],
    )


def test_gradcheck_relu_away_from_zero():
    x = Tensor(
        [
            -2.0,
            -1.0,
            1.0,
            3.0,
        ],
        requires_grad=True,
    )

    def function():
        return x.relu().sum()

    assert gradcheck(
        function,
        [x],
    )


def test_gradcheck_log_softmax():
    x = Tensor(
        [
            [0.2, -0.4, 1.2],
            [1.5, 0.1, -0.7],
        ],
        requires_grad=True,
    )

    def function():
        return (
            x.log_softmax(dim=1)
            * Tensor([
                [1.0, 2.0, 3.0],
                [0.5, -1.0, 2.0],
            ])
        ).sum()

    assert gradcheck(
        function,
        [x],
    )


def test_gradcheck_cross_entropy():
    logits = Tensor(
        [
            [0.2, -0.4, 1.2],
            [1.5, 0.1, -0.7],
        ],
        requires_grad=True,
    )

    targets = Tensor([
        2.0,
        0.0,
    ])

    criterion = CrossEntropyLoss()

    def function():
        return criterion(
            logits,
            targets,
        )

    assert gradcheck(
        function,
        [logits],
    )


def test_gradcheck_transpose_and_reshape():
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    def function():
        y = (
            x.T
            .contiguous()
            .reshape(4)
        )

        weights = Tensor([
            1.0,
            2.0,
            3.0,
            4.0,
        ])

        return (
            y * weights
        ).sum()

    assert gradcheck(
        function,
        [x],
    )


def test_gradcheck_rejects_non_scalar_output():
    x = Tensor(
        [1.0, 2.0],
        requires_grad=True,
    )

    def function():
        return x * x

    with pytest.raises(ValueError):
        gradcheck(
            function,
            [x],
        )


def test_gradcheck_requires_grad():
    x = Tensor([
        1.0,
        2.0,
    ])

    def function():
        return x.sum()

    with pytest.raises(ValueError):
        gradcheck(
            function,
            [x],
        )