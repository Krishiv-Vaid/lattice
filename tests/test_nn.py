from lattice.nn import Neuron
from lattice.value import Value


def test_neuron_parameter_count():
    neuron = Neuron(3)

    parameters = neuron.parameters()

    assert len(parameters) == 4


def test_neuron_forward():
    neuron = Neuron(2)

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
    neuron = Neuron(2)

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