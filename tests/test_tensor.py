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


def test_wrong_number_of_indices():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    with pytest.raises(IndexError):
        _ = tensor[0]


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