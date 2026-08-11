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