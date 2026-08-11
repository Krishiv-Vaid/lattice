import pytest

from lattice.value import Value


def test_value_stores_data():
    value = Value(3.0)

    assert value.data == 3.0
    assert value.grad == 0.0


def test_addition_forward():
    a = Value(2.0)
    b = Value(3.0)

    c = a + b

    assert c.data == 5.0


def test_multiplication_forward():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b

    assert c.data == 6.0


def test_addition_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a + b
    c.backward()

    assert a.grad == 1.0
    assert b.grad == 1.0


def test_multiplication_backward():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    c.backward()

    assert a.grad == 3.0
    assert b.grad == 2.0


def test_computation_graph_backward():
    x = Value(3.0)
    y = Value(4.0)

    z = x * y
    loss = z * z

    loss.backward()

    assert loss.data == 144.0
    assert x.grad == 96.0
    assert y.grad == 72.0


def test_negation():
    x = Value(3.0)

    y = -x
    y.backward()

    assert y.data == -3.0
    assert x.grad == -1.0


def test_subtraction():
    a = Value(5.0)
    b = Value(2.0)

    c = a - b
    c.backward()

    assert c.data == 3.0
    assert a.grad == 1.0
    assert b.grad == -1.0


def test_power():
    x = Value(3.0)

    y = x ** 2
    y.backward()

    assert y.data == 9.0
    assert x.grad == 6.0


def test_division():
    a = Value(6.0)
    b = Value(2.0)

    c = a / b
    c.backward()

    assert c.data == 3.0
    assert a.grad == 0.5
    assert b.grad == -1.5


def test_reverse_operations():
    x = Value(2.0)

    a = 3 + x
    b = 4 * x

    assert a.data == 5.0
    assert b.data == 8.0


def test_relu_positive():
    x = Value(3.0)

    y = x.relu()
    y.backward()

    assert y.data == 3.0
    assert x.grad == 1.0


def test_relu_negative():
    x = Value(-3.0)

    y = x.relu()
    y.backward()

    assert y.data == 0.0
    assert x.grad == 0.0


def test_exp():
    x = Value(2.0)

    y = x.exp()
    y.backward()

    assert y.data == pytest.approx(7.38905609893065)
    assert x.grad == pytest.approx(7.38905609893065)