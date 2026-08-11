from lattice.value import Value


def numerical_grad(fn, x, eps=1e-6):
    x_plus = x + eps
    x_minus = x - eps

    y_plus = fn(x_plus)
    y_minus = fn(x_minus)

    return (y_plus - y_minus) / (2 * eps)


def test_square_gradient():
    x = Value(3.0)

    y = x ** 2
    y.backward()

    analytical = x.grad

    numerical = numerical_grad(
        lambda value: value ** 2,
        3.0
    )

    assert abs(analytical - numerical) < 1e-5


def test_exp_gradient():
    x = Value(2.0)

    y = x.exp()
    y.backward()

    analytical = x.grad

    numerical = numerical_grad(
        lambda value: __import__("math").exp(value),
        2.0
    )

    assert abs(analytical - numerical) < 1e-5


def test_composed_gradient():
    x = Value(2.0)

    y = (x * x + x).exp()
    y.backward()

    analytical = x.grad

    import math

    numerical = numerical_grad(
        lambda value: math.exp(value * value + value),
        2.0
    )

    assert abs(analytical - numerical) < 1e-4