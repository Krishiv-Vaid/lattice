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