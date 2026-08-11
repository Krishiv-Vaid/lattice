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
    
def test_matmul_backward_both_operands():
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        requires_grad=True,
    )

    loss = (a @ b).sum()

    loss.backward()

    assert a.grad == [
        11.0,
        15.0,
        11.0,
        15.0,
    ]

    assert b.grad == [
        4.0,
        4.0,
        6.0,
        6.0,
    ]


def test_matmul_backward_left_only():
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    b = Tensor([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    loss = (a @ b).sum()

    loss.backward()

    assert a.grad == [
        11.0,
        15.0,
        11.0,
        15.0,
    ]

    assert b.grad is None


def test_matmul_backward_right_only():
    a = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    b = Tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        requires_grad=True,
    )

    loss = (a @ b).sum()

    loss.backward()

    assert a.grad is None

    assert b.grad == [
        4.0,
        4.0,
        6.0,
        6.0,
    ]


def test_matmul_backward_rectangular():
    a = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ],
        requires_grad=True,
    )

    loss = (a @ b).sum()

    loss.backward()

    assert a.grad == [
        15.0,
        19.0,
        23.0,
        15.0,
        19.0,
        23.0,
    ]

    assert b.grad == [
        5.0,
        5.0,
        7.0,
        7.0,
        9.0,
        9.0,
    ]


def test_matmul_backward_weighted_output():
    a = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    b = Tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        requires_grad=True,
    )

    weights = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    output = a @ b

    loss = (output * weights).sum()

    loss.backward()

    assert a.grad == [
        17.0,
        23.0,
        39.0,
        53.0,
    ]

    assert b.grad == [
        10.0,
        14.0,
        14.0,
        20.0,
    ]


def test_matmul_chain_rule():
    x = Tensor(
        [
            [1.0, 2.0],
        ],
        requires_grad=True,
    )

    w1 = Tensor(
        [
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        requires_grad=True,
    )

    w2 = Tensor(
        [
            [7.0],
            [8.0],
        ],
        requires_grad=True,
    )

    output = (x @ w1) @ w2

    loss = output.sum()

    loss.backward()

    assert x.grad == [
        53.0,
        83.0,
    ]

    assert w1.grad == [
        7.0,
        8.0,
        14.0,
        16.0,
    ]

    assert w2.grad == [
        13.0,
        16.0,
    ]


def test_matmul_with_broadcast_bias_backward():
    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    weight = Tensor(
        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0],
        ],
        requires_grad=True,
    )

    bias = Tensor(
        [1.0, 2.0, 3.0],
        requires_grad=True,
    )

    output = x @ weight + bias

    loss = output.sum()

    loss.backward()

    assert weight.grad == [
        4.0,
        4.0,
        4.0,
        6.0,
        6.0,
        6.0,
    ]

    assert bias.grad == [
        2.0,
        2.0,
        2.0,
    ]


def test_matmul_same_tensor_accumulates_gradients():
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    loss = (x @ x).sum()

    loss.backward()

    assert x.grad == [
        7.0,
        11.0,
        9.0,
        13.0,
    ]
    
def test_transpose_backward():
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    y = x.T

    loss = (y * y).sum()

    loss.backward()

    assert x.grad == [
        2.0,
        4.0,
        6.0,
        8.0,
    ]


def test_transpose_backward_preserves_index_mapping():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    weights = Tensor([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ])

    loss = (x.T * weights).sum()

    loss.backward()

    assert x.grad == [
        10.0,
        30.0,
        50.0,
        20.0,
        40.0,
        60.0,
    ]


def test_double_transpose_backward():
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    y = x.T.T

    loss = (y * 3.0).sum()

    loss.backward()

    assert x.grad == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_reshape_backward():
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    y = x.reshape(4)

    loss = (y * 3.0).sum()

    loss.backward()

    assert x.grad == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_reshape_backward_preserves_logical_order():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    y = x.reshape(3, 2)

    weights = Tensor([
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 60.0],
    ])

    loss = (y * weights).sum()

    loss.backward()

    assert x.grad == [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ]


def test_reshape_chain_backward():
    x = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    y = x.reshape(4)
    z = y.reshape(2, 2)

    loss = (z * z).sum()

    loss.backward()

    assert x.grad == [
        2.0,
        4.0,
        6.0,
        8.0,
    ]
    
def test_slice_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        requires_grad=True,
    )

    view = x[1:, 1:]

    loss = view.sum()

    loss.backward()

    assert x.grad == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
    ]


def test_slice_backward_weighted():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        requires_grad=True,
    )

    view = x[1:, 1:]

    weights = Tensor([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    loss = (view * weights).sum()

    loss.backward()

    assert x.grad == [
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
        20.0,
        0.0,
        30.0,
        40.0,
    ]


def test_row_slice_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    row = x[1]

    weights = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    loss = (row * weights).sum()

    loss.backward()

    assert x.grad == [
        0.0,
        0.0,
        0.0,
        10.0,
        20.0,
        30.0,
    ]


def test_column_slice_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        requires_grad=True,
    )

    column = x[:, 1]

    weights = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    loss = (column * weights).sum()

    loss.backward()

    assert x.grad == [
        0.0,
        10.0,
        0.0,
        0.0,
        20.0,
        0.0,
        0.0,
        30.0,
        0.0,
    ]


def test_step_slice_backward():
    x = Tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ],
        requires_grad=True,
    )

    view = x[::2]

    weights = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    loss = (view * weights).sum()

    loss.backward()

    assert x.grad == [
        10.0,
        0.0,
        20.0,
        0.0,
        30.0,
        0.0,
    ]


def test_nested_slice_backward():
    x = Tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        requires_grad=True,
    )

    first_view = x[1:, 1:]

    second_view = first_view[1:, :2]

    loss = second_view.sum()

    loss.backward()

    assert x.grad == [
        0.0,
        0.0,
        0.0,
        0.0,

        0.0,
        0.0,
        0.0,
        0.0,

        0.0,
        1.0,
        1.0,
        0.0,

        0.0,
        1.0,
        1.0,
        0.0,
    ]