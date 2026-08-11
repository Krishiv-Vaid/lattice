import random

from lattice.value import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    def __init__(self, num_inputs, nonlinearity=True):
        self.weights = [
            Value(random.uniform(-1.0, 1.0))
            for _ in range(num_inputs)
        ]

        self.bias = Value(0.0)
        self.nonlinearity = nonlinearity

    def __call__(self, inputs):
        activation = self.bias

        for weight, input_value in zip(self.weights, inputs):
            activation = activation + weight * input_value

        if self.nonlinearity:
            return activation.relu()

        return activation

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, num_inputs, num_outputs, nonlinearity=True):
        self.neurons = [
            Neuron(
                num_inputs,
                nonlinearity=nonlinearity
            )
            for _ in range(num_outputs)
        ]

    def __call__(self, inputs):
        outputs = [
            neuron(inputs)
            for neuron in self.neurons
        ]

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def parameters(self):
        parameters = []

        for neuron in self.neurons:
            parameters.extend(neuron.parameters())

        return parameters


class MLP(Module):
    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes

        self.layers = []

        for index in range(len(layer_sizes)):
            is_last_layer = index == len(layer_sizes) - 1

            layer = Layer(
                sizes[index],
                sizes[index + 1],
                nonlinearity=not is_last_layer
            )

            self.layers.append(layer)

    def __call__(self, inputs):
        output = inputs

        for layer in self.layers:
            if isinstance(output, Value):
                output = [output]

            output = layer(output)

        return output

    def parameters(self):
        parameters = []

        for layer in self.layers:
            parameters.extend(layer.parameters())

        return parameters