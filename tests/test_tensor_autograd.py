import pytest

from lattice.tensor import Tensor


def test_requires_grad_defaults_false():
    tensor = Tensor([
        1.0,
        2.0,
    ])

    assert tensor.requires_grad is False
    assert tensor.grad is None


def test_requires_grad_allocates_gradient():
    tensor = Tensor(
        [1.0, 2.0],
        requires_grad=True,
    )

    assert tensor.grad == [
        0.0,
        0.0,
    ]


def test_addition_backward():
    a = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    b = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    loss = (a + b).sum()

    loss.backward()

    assert a.grad == [
        1.0,
        1.0,
        1.0,
    ]

    assert b.grad == [
        1.0,
        1.0,
        1.0,
    ]


def test_multiplication_backward():
    a = Tensor(
        [2.0, 3.0, 4.0],
        requires_grad=True,
    )

    b = Tensor(
        [5.0, 6.0, 7.0],
        requires_grad=True,
    )

    loss = (a * b).sum()

    loss.backward()

    assert a.grad == [
        5.0,
        6.0,
        7.0,
    ]

    assert b.grad == [
        2.0,
        3.0,
        4.0,
    ]


def test_square_backward():
    x = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    loss = (x * x).sum()

    loss.backward()

    assert x.grad == [
        2.0,
        4.0,
        6.0,
    ]


def test_scalar_multiplication_backward():
    x = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    loss = (x * 5.0).sum()

    loss.backward()

    assert x.grad == [
        5.0,
        5.0,
        5.0,
    ]


def test_chain_rule():
    x = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    y = x * 2.0
    z = y * y

    loss = z.sum()

    loss.backward()

    assert x.grad == [
        8.0,
        16.0,
        24.0,
    ]


def test_zero_grad():
    x = Tensor(
        [1.0, 2.0],
        requires_grad=True,
    )

    loss = (x * x).sum()
    loss.backward()

    assert x.grad == [
        2.0,
        4.0,
    ]

    x.zero_grad()

    assert x.grad == [
        0.0,
        0.0,
    ]


def test_backward_requires_scalar():
    x = Tensor(
        [1.0, 2.0],
        requires_grad=True,
    )

    with pytest.raises(ValueError):
        x.backward()


def test_backward_requires_grad():
    x = Tensor(5.0)

    with pytest.raises(ValueError):
        x.backward()


def test_broadcast_backward_not_implemented_yet():
    a = Tensor(
        [[1.0, 2.0, 3.0]],
        requires_grad=True,
    )

    b = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    with pytest.raises(NotImplementedError):
        _ = a + b