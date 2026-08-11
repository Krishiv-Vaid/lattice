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


def test_broadcast_addition_backward_vector():
    matrix = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    vector = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    loss = (matrix + vector).sum()

    loss.backward()

    assert matrix.grad == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    assert vector.grad == [
        2.0,
        2.0,
        2.0,
    ]


def test_broadcast_addition_backward_column():
    matrix = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    column = Tensor(
        [
            [10.0],
            [20.0],
        ],
        requires_grad=True,
    )

    loss = (matrix + column).sum()

    loss.backward()

    assert matrix.grad == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    assert column.grad == [
        3.0,
        3.0,
    ]


def test_broadcast_multiplication_backward_vector():
    matrix = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    vector = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    loss = (matrix * vector).sum()

    loss.backward()

    assert matrix.grad == [
        10.0,
        20.0,
        30.0,
        10.0,
        20.0,
        30.0,
    ]

    assert vector.grad == [
        5.0,
        7.0,
        9.0,
    ]


def test_broadcast_multiplication_backward_column():
    matrix = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    column = Tensor(
        [
            [10.0],
            [20.0],
        ],
        requires_grad=True,
    )

    loss = (matrix * column).sum()

    loss.backward()

    assert matrix.grad == [
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
    ]

    assert column.grad == [
        6.0,
        15.0,
    ]


def test_scalar_tensor_broadcast_backward():
    matrix = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    scalar = Tensor(
        10.0,
        requires_grad=True,
    )

    loss = (matrix * scalar).sum()

    loss.backward()

    assert matrix.grad == [
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]

    assert scalar.grad == [
        21.0,
    ]


def test_broadcast_chain_rule():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    bias = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    y = x + bias
    z = y * 2.0

    loss = z.sum()

    loss.backward()

    assert x.grad == [
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ]

    assert bias.grad == [
        4.0,
        4.0,
        4.0,
    ]


def test_multidimensional_broadcast_backward():
    a = Tensor(
        [
            [
                [1.0, 2.0, 3.0],
            ],
            [
                [4.0, 5.0, 6.0],
            ],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [10.0],
            [20.0],
            [30.0],
            [40.0],
        ],
        requires_grad=True,
    )

    loss = (a + b).sum()

    loss.backward()

    assert a.grad == [
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
    ]

    assert b.grad == [
        6.0,
        6.0,
        6.0,
        6.0,
    ]
    
def test_sum_dim_zero_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    reduced = x.sum(dim=0)

    weights = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    loss = (reduced * weights).sum()

    loss.backward()

    assert x.grad == [
        10.0,
        20.0,
        30.0,
        10.0,
        20.0,
        30.0,
    ]


def test_sum_dim_one_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    reduced = x.sum(dim=1)

    weights = Tensor([
        10.0,
        20.0,
    ])

    loss = (reduced * weights).sum()

    loss.backward()

    assert x.grad == [
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
    ]


def test_mean_dim_zero_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    loss = x.mean(dim=0).sum()

    loss.backward()

    assert x.grad == [
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    ]


def test_mean_dim_one_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    loss = x.mean(dim=1).sum()

    loss.backward()

    expected = 1.0 / 3.0

    assert x.grad == [
        expected,
        expected,
        expected,
        expected,
        expected,
        expected,
    ]


def test_negative_dimension_reduction_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    loss = x.sum(dim=-1).sum()

    loss.backward()

    assert x.grad == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_three_dimensional_reduction_backward():
    x = Tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ],
        requires_grad=True,
    )

    reduced = x.sum(dim=1)

    weights = Tensor([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    loss = (reduced * weights).sum()

    loss.backward()

    assert x.grad == [
        10.0,
        20.0,
        10.0,
        20.0,
        30.0,
        40.0,
        30.0,
        40.0,
    ]


def test_one_dimensional_sum_dim_backward():
    x = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    loss = x.sum(dim=0)

    assert loss.shape == ()

    loss.backward()

    assert x.grad == [
        1.0,
        1.0,
        1.0,
    ]


def test_reduction_backward_chain_rule():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    row_sums = x.sum(dim=1)

    squared = row_sums * row_sums

    loss = squared.sum()

    loss.backward()

    assert x.grad == [
        12.0,
        12.0,
        12.0,
        30.0,
        30.0,
        30.0,
    ]