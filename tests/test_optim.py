from lattice.optim import SGD
from lattice.value import Value


def test_sgd_step():
    parameter = Value(5.0)

    parameter.grad = 2.0

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    optimizer.step()

    assert parameter.data == 4.8


def test_sgd_zero_grad():
    parameter = Value(5.0)

    parameter.grad = 7.0

    optimizer = SGD(
        [parameter],
        lr=0.1
    )

    optimizer.zero_grad()

    assert parameter.grad == 0.0