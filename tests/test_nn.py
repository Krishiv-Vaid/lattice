from lattice.nn import Layer, MLP, MSELoss, Module, Neuron
from lattice.value import Value
from lattice.tensor import Tensor
from lattice.nn import Linear

def test_module_has_no_parameters_by_default():
    module = Module()

    assert module.parameters() == []


def test_neuron_parameter_count():
    neuron = Neuron(3)

    parameters = neuron.parameters()

    assert len(parameters) == 4


def test_neuron_forward():
    neuron = Neuron(
        2,
        nonlinearity=False
    )

    neuron.weights[0].data = 2.0
    neuron.weights[1].data = 3.0
    neuron.bias.data = 1.0

    inputs = [
        Value(4.0),
        Value(5.0),
    ]

    output = neuron(inputs)

    assert output.data == 24.0


def test_neuron_backward():
    neuron = Neuron(
        2,
        nonlinearity=False
    )

    neuron.weights[0].data = 2.0
    neuron.weights[1].data = 3.0
    neuron.bias.data = 1.0

    x1 = Value(4.0)
    x2 = Value(5.0)

    output = neuron([x1, x2])

    output.backward()

    assert neuron.weights[0].grad == 4.0
    assert neuron.weights[1].grad == 5.0
    assert neuron.bias.grad == 1.0

    assert x1.grad == 2.0
    assert x2.grad == 3.0


def test_neuron_relu():
    neuron = Neuron(
        1,
        nonlinearity=True
    )

    neuron.weights[0].data = -2.0
    neuron.bias.data = 0.0

    output = neuron([Value(3.0)])

    assert output.data == 0.0


def test_layer_output_count():
    layer = Layer(
        num_inputs=2,
        num_outputs=3,
        nonlinearity=False
    )

    inputs = [
        Value(1.0),
        Value(2.0),
    ]

    outputs = layer(inputs)

    assert len(outputs) == 3


def test_layer_parameter_count():
    layer = Layer(
        num_inputs=2,
        num_outputs=3
    )

    assert len(layer.parameters()) == 9


def test_mlp_structure():
    model = MLP(
        num_inputs=2,
        layer_sizes=[4, 4, 1]
    )

    assert len(model.layers) == 3

    assert len(model.layers[0].neurons) == 4
    assert len(model.layers[1].neurons) == 4
    assert len(model.layers[2].neurons) == 1


def test_mlp_forward():
    model = MLP(
        num_inputs=2,
        layer_sizes=[3, 1]
    )

    inputs = [
        Value(1.0),
        Value(2.0),
    ]

    output = model(inputs)

    assert isinstance(output, Value)


def test_zero_grad():
    neuron = Neuron(
        2,
        nonlinearity=False
    )

    x1 = Value(2.0)
    x2 = Value(3.0)

    output = neuron([x1, x2])
    output.backward()

    assert any(
        parameter.grad != 0.0
        for parameter in neuron.parameters()
    )

    neuron.zero_grad()

    assert all(
        parameter.grad == 0.0
        for parameter in neuron.parameters()
    )


def test_mse_loss():
    loss_fn = MSELoss()

    predictions = [
        Value(2.0),
        Value(4.0),
    ]

    targets = [
        3.0,
        6.0,
    ]

    loss = loss_fn(
        predictions,
        targets
    )

    assert loss.data == 2.5


def test_mse_loss_backward():
    loss_fn = MSELoss()

    prediction = Value(3.0)

    loss = loss_fn(
        [prediction],
        [5.0]
    )

    loss.backward()

    assert prediction.grad == -4.0
    
def test_tensor_linear_parameters():
    linear = Linear(
        2,
        3,
    )

    parameters = linear.parameters()

    assert len(parameters) == 2

    assert parameters[0] is linear.weight
    assert parameters[1] is linear.bias

    assert linear.weight.shape == (2, 3)
    assert linear.bias.shape == (3,)

    assert linear.weight.requires_grad
    assert linear.bias.requires_grad


def test_tensor_linear_forward():
    linear = Linear(
        2,
        3,
    )

    linear.weight = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    linear.bias = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    output = linear(x)

    assert output.shape == (2, 3)

    assert output.data == [
        19.0,
        32.0,
        45.0,
        29.0,
        46.0,
        63.0,
    ]


def test_tensor_linear_backward():
    linear = Linear(
        2,
        3,
    )

    linear.weight = Tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )

    linear.bias = Tensor(
        [10.0, 20.0, 30.0],
        requires_grad=True,
    )

    x = Tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    loss = linear(x).sum()

    loss.backward()

    assert linear.weight.grad == [
        4.0,
        4.0,
        4.0,
        6.0,
        6.0,
        6.0,
    ]

    assert linear.bias.grad == [
        2.0,
        2.0,
        2.0,
    ]


def test_tensor_linear_without_bias():
    linear = Linear(
        2,
        2,
        bias=False,
    )

    linear.weight = Tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        requires_grad=True,
    )

    x = Tensor([
        [1.0, 2.0],
    ])

    output = linear(x)

    assert linear.bias is None
    assert len(linear.parameters()) == 1

    assert output.data == [
        7.0,
        10.0,
    ]


def test_tensor_module_zero_grad():
    linear = Linear(
        2,
        1,
    )

    linear.weight = Tensor(
        [
            [1.0],
            [2.0],
        ],
        requires_grad=True,
    )

    linear.bias = Tensor(
        [0.0],
        requires_grad=True,
    )

    x = Tensor([
        [1.0, 2.0],
    ])

    loss = linear(x).sum()

    loss.backward()

    assert linear.weight.grad != [
        0.0,
        0.0,
    ]

    assert linear.bias.grad != [
        0.0,
    ]

    linear.zero_grad()

    assert linear.weight.grad == [
        0.0,
        0.0,
    ]

    assert linear.bias.grad == [
        0.0,
    ]