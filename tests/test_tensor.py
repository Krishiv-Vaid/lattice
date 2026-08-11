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

    assert tensor.data == [1.0, 2.0, 3.0]
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
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
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
    try:
        Tensor([
            [1.0, 2.0],
            [3.0],
        ])
    except ValueError:
        return

    raise AssertionError(
        "Tensor should reject non-rectangular data"
    )


def test_repr():
    tensor = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result = repr(tensor)

    assert "shape=(2, 2)" in result
    assert "strides=(2, 1)" in result