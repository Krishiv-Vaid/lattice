import random

from lattice.value import Value


class Neuron:
    def __init__(self, num_inputs):
        self.weights = [
            Value(random.uniform(-1.0, 1.0))
            for _ in range(num_inputs)
        ]

        self.bias = Value(0.0)

    def __call__(self, inputs):
        activation = self.bias

        for weight, input_value in zip(self.weights, inputs):
            activation = activation + weight * input_value

        return activation

    def parameters(self):
        return self.weights + [self.bias]