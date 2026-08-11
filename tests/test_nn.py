from lattice.nn import Layer, MLP, MSELoss, Module, Neuron
from lattice.value import Value


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