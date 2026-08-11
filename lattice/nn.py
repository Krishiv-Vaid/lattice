import random

from lattice.tensor import Tensor
from lattice.value import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            if hasattr(parameter, "zero_grad"):
                parameter.zero_grad()
            else:
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


class Linear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
    ):
        if not isinstance(in_features, int):
            raise TypeError(
                "in_features must be an integer"
            )

        if not isinstance(out_features, int):
            raise TypeError(
                "out_features must be an integer"
            )

        if in_features <= 0:
            raise ValueError(
                "in_features must be positive"
            )

        if out_features <= 0:
            raise ValueError(
                "out_features must be positive"
            )

        self.in_features = in_features
        self.out_features = out_features

        scale = 1.0 / (in_features ** 0.5)

        weight_data = [
            [
                random.uniform(
                    -scale,
                    scale,
                )
                for _ in range(out_features)
            ]
            for _ in range(in_features)
        ]

        self.weight = Tensor(
            weight_data,
            requires_grad=True,
        )

        if bias:
            self.bias = Tensor(
                [0.0] * out_features,
                requires_grad=True,
            )
        else:
            self.bias = None

    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise TypeError(
                "Linear input must be a Tensor"
            )

        if x.ndim != 2:
            raise ValueError(
                "Linear currently expects a 2D Tensor"
            )

        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input with "
                f"{self.in_features} features, "
                f"got {x.shape[1]}"
            )

        output = x @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return output

    def parameters(self):
        parameters = [self.weight]

        if self.bias is not None:
            parameters.append(self.bias)

        return parameters


class ReLU(Module):
    def __call__(self, x):
        if not isinstance(x, Tensor):
            raise TypeError(
                "ReLU input must be a Tensor"
            )

        return x.relu()

    def parameters(self):
        return []

class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

        for module in self.modules:
            if not isinstance(module, Module):
                raise TypeError(
                    "Sequential expects Module objects"
                )

    def __call__(self, x):
        output = x

        for module in self.modules:
            output = module(output)

        return output

    def parameters(self):
        parameters = []

        for module in self.modules:
            parameters.extend(
                module.parameters()
            )

        return parameters

    def __len__(self):
        return len(self.modules)

    def __getitem__(self, index):
        return self.modules[index]

class TensorMSELoss(Module):
    def __call__(self, prediction, target):
        if not isinstance(prediction, Tensor):
            raise TypeError(
                "prediction must be a Tensor"
            )

        if not isinstance(target, Tensor):
            raise TypeError(
                "target must be a Tensor"
            )

        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction shape "
                f"{prediction.shape} does not match "
                f"target shape {target.shape}"
            )

        difference = prediction - target

        return (
            difference
            * difference
        ).mean()

    def parameters(self):
        return []
    
class CrossEntropyLoss(Module):
    def __call__(self, logits, targets):
        if not isinstance(logits, Tensor):
            raise TypeError(
                "logits must be a Tensor"
            )

        if not isinstance(targets, Tensor):
            raise TypeError(
                "targets must be a Tensor"
            )

        if logits.ndim != 2:
            raise ValueError(
                "CrossEntropyLoss expects "
                "2D logits"
            )

        if targets.ndim != 1:
            raise ValueError(
                "CrossEntropyLoss expects "
                "1D class-index targets"
            )

        batch_size, num_classes = logits.shape

        if targets.shape[0] != batch_size:
            raise ValueError(
                "targets batch size must match logits"
            )

        log_probs = logits.log_softmax(dim=1)

        total = 0.0

        for batch_index in range(batch_size):
            class_index = int(
                targets[batch_index]
            )

            if (
                class_index < 0
                or class_index >= num_classes
            ):
                raise ValueError(
                    "target class index out of range"
                )

            total -= log_probs[
                batch_index,
                class_index,
            ]

        loss_value = total / batch_size

        out = Tensor(
            loss_value,
            requires_grad=logits.requires_grad,
            _children=(log_probs,),
            _op="cross_entropy",
        )

        if logits.requires_grad:
            def _backward():
                upstream = out.grad[0]

                for batch_index in range(
                    batch_size
                ):
                    class_index = int(
                        targets[batch_index]
                    )

                    flat = (
                        log_probs._flat_logical_index(
                            (
                                batch_index,
                                class_index,
                            )
                        )
                    )

                    log_probs.grad[flat] += (
                        -upstream / batch_size
                    )

            out._backward = _backward

        return out

    def parameters(self):
        return []


class Layer(Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        nonlinearity=True
    ):
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
            is_last_layer = (
                index == len(layer_sizes) - 1
            )

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


class MSELoss:
    def __call__(self, predictions, targets):
        if len(predictions) != len(targets):
            raise ValueError(
                "predictions and targets must have the same length"
            )

        losses = [
            (prediction - target) ** 2
            for prediction, target in zip(
                predictions,
                targets
            )
        ]

        return sum(losses) / len(losses)


__all__ = [
    "Module",
    "Linear",
    "ReLU",
    "Sequential",
    "TensorMSELoss",
    "Neuron",
    "Layer",
    "MLP",
    "MSELoss",
]