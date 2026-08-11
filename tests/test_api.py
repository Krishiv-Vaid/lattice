import lattice
import lattice.nn as nn
import lattice.optim as optim


def test_lattice_version():
    assert lattice.__version__ == "0.1.0"


def test_value_public_api():
    value = lattice.Value(3.0)

    assert value.data == 3.0


def test_nn_public_api():
    model = nn.MLP(
        num_inputs=2,
        layer_sizes=[4, 1]
    )

    assert isinstance(model, nn.Module)


def test_optim_public_api():
    parameter = lattice.Value(1.0)

    optimizer = optim.SGD(
        [parameter],
        lr=0.01
    )

    assert optimizer.lr == 0.01