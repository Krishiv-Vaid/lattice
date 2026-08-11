import pytest

torch = pytest.importorskip("torch")

from lattice.nn import (
    CrossEntropyLoss,
    Linear,
    ReLU,
    Sequential,
)
from lattice.tensor import Tensor


def assert_close(
    actual,
    expected,
    *,
    rel=1e-6,
    abs=1e-6,
):
    assert actual == pytest.approx(
        expected,
        rel=rel,
        abs=abs,
    )


def make_lattice_model():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 2),
    )

    model[0].weight = Tensor(
        [
            [0.2, -0.3, 0.5],
            [0.7, 0.1, -0.4],
        ],
        requires_grad=True,
    )

    model[0].bias = Tensor(
        [0.1, -0.2, 0.3],
        requires_grad=True,
    )

    model[2].weight = Tensor(
        [
            [0.6, -0.1],
            [-0.5, 0.8],
            [0.2, 0.4],
        ],
        requires_grad=True,
    )

    model[2].bias = Tensor(
        [-0.1, 0.2],
        requires_grad=True,
    )

    return model


def make_torch_parameters():
    w1 = torch.tensor(
        [
            [0.2, -0.3, 0.5],
            [0.7, 0.1, -0.4],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    b1 = torch.tensor(
        [0.1, -0.2, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )

    w2 = torch.tensor(
        [
            [0.6, -0.1],
            [-0.5, 0.8],
            [0.2, 0.4],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    b2 = torch.tensor(
        [-0.1, 0.2],
        dtype=torch.float64,
        requires_grad=True,
    )

    return w1, b1, w2, b2


def torch_forward(
    x,
    w1,
    b1,
    w2,
    b2,
):
    hidden = torch.relu(
        x @ w1 + b1
    )

    return hidden @ w2 + b2


def test_linear_relu_linear_forward_matches_pytorch():
    x_data = [
        [1.0, 2.0],
        [-1.0, 0.5],
        [0.25, -0.75],
    ]

    lattice_model = make_lattice_model()

    lattice_x = Tensor(
        x_data,
        requires_grad=True,
    )

    lattice_logits = lattice_model(
        lattice_x
    )

    w1, b1, w2, b2 = (
        make_torch_parameters()
    )

    torch_x = torch.tensor(
        x_data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_logits = torch_forward(
        torch_x,
        w1,
        b1,
        w2,
        b2,
    )

    assert lattice_logits.shape == (
        *torch_logits.shape,
    )

    assert_close(
        lattice_logits.data,
        torch_logits.detach()
        .reshape(-1)
        .tolist(),
    )


def test_cross_entropy_forward_matches_pytorch():
    x_data = [
        [1.0, 2.0],
        [-1.0, 0.5],
        [0.25, -0.75],
    ]

    targets_data = [
        0.0,
        1.0,
        0.0,
    ]

    lattice_model = make_lattice_model()

    lattice_logits = lattice_model(
        Tensor(x_data)
    )

    lattice_loss = CrossEntropyLoss()(
        lattice_logits,
        Tensor(targets_data),
    )

    w1, b1, w2, b2 = (
        make_torch_parameters()
    )

    torch_x = torch.tensor(
        x_data,
        dtype=torch.float64,
    )

    torch_targets = torch.tensor(
        [0, 1, 0],
        dtype=torch.long,
    )

    torch_logits = torch_forward(
        torch_x,
        w1,
        b1,
        w2,
        b2,
    )

    torch_loss = (
        torch.nn.functional.cross_entropy(
            torch_logits,
            torch_targets,
        )
    )

    assert_close(
        lattice_loss.data[0],
        torch_loss.item(),
    )


def test_network_parameter_gradients_match_pytorch():
    x_data = [
        [1.0, 2.0],
        [-1.0, 0.5],
        [0.25, -0.75],
    ]

    targets_data = [
        0.0,
        1.0,
        0.0,
    ]

    lattice_model = make_lattice_model()

    lattice_x = Tensor(
        x_data,
        requires_grad=True,
    )

    lattice_loss = CrossEntropyLoss()(
        lattice_model(lattice_x),
        Tensor(targets_data),
    )

    lattice_loss.backward()

    w1, b1, w2, b2 = (
        make_torch_parameters()
    )

    torch_x = torch.tensor(
        x_data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_targets = torch.tensor(
        [0, 1, 0],
        dtype=torch.long,
    )

    torch_logits = torch_forward(
        torch_x,
        w1,
        b1,
        w2,
        b2,
    )

    torch_loss = (
        torch.nn.functional.cross_entropy(
            torch_logits,
            torch_targets,
        )
    )

    torch_loss.backward()

    assert_close(
        lattice_model[0].weight.grad,
        w1.grad.reshape(-1).tolist(),
    )

    assert_close(
        lattice_model[0].bias.grad,
        b1.grad.tolist(),
    )

    assert_close(
        lattice_model[2].weight.grad,
        w2.grad.reshape(-1).tolist(),
    )

    assert_close(
        lattice_model[2].bias.grad,
        b2.grad.tolist(),
    )


def test_network_input_gradients_match_pytorch():
    x_data = [
        [1.0, 2.0],
        [-1.0, 0.5],
        [0.25, -0.75],
    ]

    targets_data = [
        0.0,
        1.0,
        0.0,
    ]

    lattice_model = make_lattice_model()

    lattice_x = Tensor(
        x_data,
        requires_grad=True,
    )

    lattice_loss = CrossEntropyLoss()(
        lattice_model(lattice_x),
        Tensor(targets_data),
    )

    lattice_loss.backward()

    w1, b1, w2, b2 = (
        make_torch_parameters()
    )

    torch_x = torch.tensor(
        x_data,
        dtype=torch.float64,
        requires_grad=True,
    )

    torch_targets = torch.tensor(
        [0, 1, 0],
        dtype=torch.long,
    )

    torch_loss = (
        torch.nn.functional.cross_entropy(
            torch_forward(
                torch_x,
                w1,
                b1,
                w2,
                b2,
            ),
            torch_targets,
        )
    )

    torch_loss.backward()

    assert_close(
        lattice_x.grad,
        torch_x.grad.reshape(-1).tolist(),
    )