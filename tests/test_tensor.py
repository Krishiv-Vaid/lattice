import pytest

from lattice.tensor import Tensor


def test_scalar_tensor():
    tensor = Tensor(5.0)

    assert tensor.data == [5.0]
    assert tensor.shape == ()
    assert tensor.strides == ()
    assert tensor.ndim == 0
    assert tensor.numel == 1


def test_vector_tensor():
    tensor = Tensor([1.0, 2.0, 3.0])

    assert tensor.data == [
        1.0,
        2.0,
        3.0,
    ]

    assert tensor.shape == (3,)
    assert tensor.strides == (1,)
    assert tensor.ndim == 1
    assert tensor.numel == 3


def test_matrix_tensor():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    assert tensor.data == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]

    assert tensor.shape == (2, 3)
    assert tensor.strides == (3, 1)
    assert tensor.ndim == 2
    assert tensor.numel == 6


def test_three_dimensional_tensor():
    tensor = Tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
    ])

    assert tensor.shape == (2, 2, 2)
    assert tensor.strides == (4, 2, 1)
    assert tensor.numel == 8


def test_rejects_ragged_data():
    with pytest.raises(ValueError):
        Tensor([
            [1.0, 2.0],
            [3.0],
        ])


def test_repr():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result = repr(tensor)

    assert "shape=(2, 2)" in result
    assert "strides=(2, 1)" in result
    assert "offset=0" in result


def test_vector_indexing():
    tensor = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    assert tensor[0] == 10.0
    assert tensor[1] == 20.0
    assert tensor[2] == 30.0


def test_matrix_indexing():
    tensor = Tensor([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
    ])

    assert tensor[0, 0] == 10.0
    assert tensor[0, 2] == 30.0
    assert tensor[1, 0] == 40.0
    assert tensor[1, 2] == 60.0


def test_three_dimensional_indexing():
    tensor = Tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
    ])

    assert tensor[0, 0, 0] == 1.0
    assert tensor[0, 1, 1] == 4.0
    assert tensor[1, 0, 1] == 6.0
    assert tensor[1, 1, 1] == 8.0


def test_negative_indexing():
    tensor = Tensor([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    assert tensor[-1, -1] == 40.0
    assert tensor[-2, -2] == 10.0


def test_index_out_of_range():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(IndexError):
        _ = tensor[2, 0]


def test_too_many_indices():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(IndexError):
        _ = tensor[0, 0, 0]


def test_non_integer_index():
    tensor = Tensor([
        1.0,
        2.0,
    ])

    with pytest.raises(TypeError):
        _ = tensor[1.5]


def test_tensor_assignment():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    tensor[1, 0] = 99.0

    assert tensor[1, 0] == 99.0

    assert tensor.data == [
        1.0,
        2.0,
        99.0,
        4.0,
    ]


def test_transpose_metadata():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.transpose(0, 1)

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert transposed.offset == 0


def test_transpose_indexing():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.transpose(0, 1)

    assert transposed[0, 0] == 1.0
    assert transposed[0, 1] == 4.0
    assert transposed[1, 0] == 2.0
    assert transposed[1, 1] == 5.0
    assert transposed[2, 0] == 3.0
    assert transposed[2, 1] == 6.0


def test_transpose_shares_storage():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    transposed = tensor.transpose(0, 1)

    transposed[0, 1] = 99.0

    assert tensor[1, 0] == 99.0

    tensor[0, 1] = 123.0

    assert transposed[1, 0] == 123.0

    assert tensor.data is transposed.data


def test_T_property():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.T

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)
    assert transposed[2, 1] == 6.0


def test_three_dimensional_transpose():
    tensor = Tensor([
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ],
    ])

    assert tensor.shape == (2, 3, 2)
    assert tensor.strides == (6, 2, 1)

    transposed = tensor.transpose(0, 1)

    assert transposed.shape == (3, 2, 2)
    assert transposed.strides == (2, 6, 1)

    assert transposed[0, 0, 0] == 1.0
    assert transposed[0, 1, 0] == 7.0
    assert transposed[2, 1, 1] == 12.0


def test_invalid_transpose_dimension():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(IndexError):
        tensor.transpose(0, 2)


def test_negative_transpose_dimensions():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.transpose(-1, -2)

    assert transposed.shape == (3, 2)
    assert transposed.strides == (1, 3)


def test_contiguous_tensor():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    assert tensor.is_contiguous


def test_transpose_is_not_contiguous():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.T

    assert not transposed.is_contiguous


def test_reshape_contiguous_tensor():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    reshaped = tensor.reshape(3, 2)

    assert reshaped.shape == (3, 2)
    assert reshaped.strides == (2, 1)

    assert reshaped[0, 0] == 1.0
    assert reshaped[1, 0] == 3.0
    assert reshaped[2, 1] == 6.0

    assert reshaped.data is tensor.data


def test_reshape_accepts_tuple():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    reshaped = tensor.reshape((2, 2))

    assert reshaped.shape == (2, 2)
    assert reshaped[1, 1] == 4.0


def test_reshape_rejects_wrong_numel():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    with pytest.raises(ValueError):
        tensor.reshape(3, 2)


def test_reshape_rejects_non_contiguous_tensor():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.T

    with pytest.raises(ValueError):
        transposed.reshape(6)


def test_contiguous_returns_same_tensor_when_possible():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    contiguous = tensor.contiguous()

    assert contiguous is tensor


def test_contiguous_copies_transposed_tensor():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.T

    contiguous = transposed.contiguous()

    assert contiguous.shape == (3, 2)
    assert contiguous.strides == (2, 1)
    assert contiguous.is_contiguous

    assert contiguous.data == [
        1.0,
        4.0,
        2.0,
        5.0,
        3.0,
        6.0,
    ]

    assert contiguous.data is not tensor.data


def test_reshape_after_contiguous():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    result = (
        tensor
        .T
        .contiguous()
        .reshape(2, 3)
    )

    assert result.shape == (2, 3)

    assert result.data == [
        1.0,
        4.0,
        2.0,
        5.0,
        3.0,
        6.0,
    ]
    
def test_row_slice():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    row = tensor[1, :]

    assert row.shape == (3,)
    assert row.strides == (1,)
    assert row.offset == 3

    assert row[0] == 4.0
    assert row[1] == 5.0
    assert row[2] == 6.0


def test_column_slice():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    column = tensor[:, 1]

    assert column.shape == (2,)
    assert column.strides == (3,)
    assert column.offset == 1

    assert column[0] == 2.0
    assert column[1] == 5.0


def test_submatrix_slice():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    view = tensor[1:, 1:]

    assert view.shape == (2, 2)
    assert view.strides == (3, 1)
    assert view.offset == 4

    assert view[0, 0] == 5.0
    assert view[0, 1] == 6.0
    assert view[1, 0] == 8.0
    assert view[1, 1] == 9.0


def test_slice_shares_storage():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    row = tensor[1, :]

    assert row.data is tensor.data

    row[0] = 99.0

    assert tensor[1, 0] == 99.0


def test_slice_with_step():
    tensor = Tensor([
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
    ])

    view = tensor[::2]

    assert view.shape == (3,)
    assert view.strides == (2,)
    assert view.offset == 0

    assert view[0] == 10.0
    assert view[1] == 30.0
    assert view[2] == 50.0


def test_slice_with_start_and_stop():
    tensor = Tensor([
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ])

    view = tensor[1:4]

    assert view.shape == (3,)
    assert view.strides == (1,)
    assert view.offset == 1

    assert view[0] == 20.0
    assert view[2] == 40.0


def test_missing_dimensions_become_full_slices():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    view = tensor[1]

    assert isinstance(view, Tensor)

    assert view.shape == (3,)
    assert view.offset == 3

    assert view[0] == 4.0
    assert view[2] == 6.0


def test_slice_of_transpose():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = tensor.T

    view = transposed[1:, :]

    assert view.shape == (2, 2)

    assert view[0, 0] == 2.0
    assert view[0, 1] == 5.0
    assert view[1, 0] == 3.0
    assert view[1, 1] == 6.0

    assert view.data is tensor.data


def test_strided_slice_is_not_contiguous():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ])

    view = tensor[::2]

    assert not view.is_contiguous


def test_contiguous_from_slice():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ])

    view = tensor[::2]

    contiguous = view.contiguous()

    assert contiguous.data == [
        1.0,
        3.0,
        5.0,
    ]

    assert contiguous.shape == (3,)
    assert contiguous.strides == (1,)
    assert contiguous.offset == 0

    assert contiguous.data is not tensor.data

def test_tensor_addition():
    a = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    b = Tensor([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    result = a + b

    assert result.shape == (2, 2)
    assert result.data == [
        11.0,
        22.0,
        33.0,
        44.0,
    ]


def test_tensor_subtraction():
    a = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    b = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    result = a - b

    assert result.data == [
        9.0,
        18.0,
        27.0,
    ]


def test_tensor_multiplication():
    a = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    b = Tensor([
        4.0,
        5.0,
        6.0,
    ])

    result = a * b

    assert result.data == [
        4.0,
        10.0,
        18.0,
    ]


def test_tensor_division():
    a = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    b = Tensor([
        2.0,
        4.0,
        5.0,
    ])

    result = a / b

    assert result.data == [
        5.0,
        5.0,
        6.0,
    ]


def test_tensor_scalar_addition():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    result = tensor + 10

    assert result.data == [
        11.0,
        12.0,
        13.0,
    ]


def test_reverse_scalar_addition():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    result = 10 + tensor

    assert result.data == [
        11.0,
        12.0,
        13.0,
    ]


def test_tensor_scalar_multiplication():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    result = tensor * 3

    assert result.data == [
        3.0,
        6.0,
        9.0,
    ]


def test_reverse_scalar_subtraction():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    result = 10 - tensor

    assert result.data == [
        9.0,
        8.0,
        7.0,
    ]


def test_reverse_scalar_division():
    tensor = Tensor([
        2.0,
        4.0,
        5.0,
    ])

    result = 20 / tensor

    assert result.data == [
        10.0,
        5.0,
        4.0,
    ]


def test_tensor_negation():
    tensor = Tensor([
        1.0,
        -2.0,
        3.0,
    ])

    result = -tensor

    assert result.data == [
        -1.0,
        2.0,
        -3.0,
    ]


def test_elementwise_shape_mismatch():
    a = Tensor([
        1.0,
        2.0,
    ])

    b = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        _ = a + b


def test_elementwise_operation_on_transpose():
    a = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    transposed = a.T

    result = transposed * 10

    assert result.shape == (3, 2)

    assert result.data == [
        10.0,
        40.0,
        20.0,
        50.0,
        30.0,
        60.0,
    ]


def test_elementwise_operation_on_slice():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    view = tensor[1:, 1:]

    result = view + 100

    assert result.shape == (2, 2)

    assert result.data == [
        105.0,
        106.0,
        108.0,
        109.0,
    ]


def test_elementwise_result_is_contiguous():
    tensor = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    result = tensor.T + 1

    assert result.is_contiguous
    assert result.offset == 0
    assert result.strides == (2, 1)
    
def test_broadcast_vector_to_matrix():
    tensor = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    broadcasted = tensor.broadcast_to(
        (2, 3)
    )

    assert broadcasted.shape == (2, 3)
    assert broadcasted.strides == (0, 1)

    assert broadcasted[0, 0] == 10.0
    assert broadcasted[0, 2] == 30.0
    assert broadcasted[1, 0] == 10.0
    assert broadcasted[1, 2] == 30.0

    assert broadcasted.data is tensor.data


def test_broadcast_singleton_dimension():
    tensor = Tensor([
        [10.0],
        [20.0],
    ])

    broadcasted = tensor.broadcast_to(
        (2, 3)
    )

    assert broadcasted.shape == (2, 3)
    assert broadcasted.strides == (1, 0)

    assert broadcasted[0, 0] == 10.0
    assert broadcasted[0, 2] == 10.0

    assert broadcasted[1, 0] == 20.0
    assert broadcasted[1, 2] == 20.0


def test_broadcast_add_vector_to_matrix():
    matrix = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    vector = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    result = matrix + vector

    assert result.shape == (2, 3)

    assert result.data == [
        11.0,
        22.0,
        33.0,
        14.0,
        25.0,
        36.0,
    ]


def test_broadcast_column_to_matrix():
    matrix = Tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    column = Tensor([
        [10.0],
        [20.0],
    ])

    result = matrix + column

    assert result.data == [
        11.0,
        12.0,
        13.0,
        24.0,
        25.0,
        26.0,
    ]


def test_broadcast_multiple_dimensions():
    a = Tensor([
        [
            [1.0, 2.0, 3.0],
        ],
        [
            [4.0, 5.0, 6.0],
        ],
    ])

    b = Tensor([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
    ])

    result = a + b

    assert result.shape == (2, 4, 3)

    assert result[0, 0, 0] == 11.0
    assert result[0, 3, 2] == 43.0
    assert result[1, 0, 0] == 14.0
    assert result[1, 3, 2] == 46.0


def test_broadcast_scalar_shaped_tensor():
    scalar = Tensor(10.0)

    matrix = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result = matrix + scalar

    assert result.shape == (2, 2)

    assert result.data == [
        11.0,
        12.0,
        13.0,
        14.0,
    ]


def test_incompatible_broadcast_shapes():
    a = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    b = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        _ = a + b


def test_broadcast_to_invalid_shape():
    tensor = Tensor([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        tensor.broadcast_to(
            (2, 2)
        )


def test_broadcasted_stride_zero_reuses_storage():
    tensor = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    view = tensor.broadcast_to(
        (2, 3)
    )

    tensor[1] = 99.0

    assert view[0, 1] == 99.0
    assert view[1, 1] == 99.0


def test_broadcast_with_transpose():
    matrix = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    transposed = matrix.T

    vector = Tensor([
        10.0,
        20.0,
        30.0,
    ])

    result = transposed + vector

    assert result.shape == (2, 3)

    assert result.data == [
        11.0,
        23.0,
        35.0,
        12.0,
        24.0,
        36.0,
    ]