import math

from lattice.tensor import Tensor
from lattice.value import Value


class SGD:
    def __init__(
        self,
        parameters,
        lr=0.01,
    ):
        self.parameters = list(parameters)
        self.lr = lr

        if self.lr <= 0:
            raise ValueError(
                "learning rate must be positive"
            )

    def zero_grad(self):
        for parameter in self.parameters:
            if isinstance(parameter, Tensor):
                parameter.zero_grad()

            elif isinstance(parameter, Value):
                parameter.grad = 0.0

            else:
                raise TypeError(
                    "SGD parameters must be "
                    "Tensor or Value objects"
                )

    def step(self):
        for parameter in self.parameters:
            if isinstance(parameter, Tensor):
                if not parameter.requires_grad:
                    continue

                if parameter.grad is None:
                    continue

                for index in range(
                    parameter.numel
                ):
                    parameter.data[index] -= (
                        self.lr
                        * parameter.grad[index]
                    )

            elif isinstance(parameter, Value):
                parameter.data -= (
                    self.lr
                    * parameter.grad
                )

            else:
                raise TypeError(
                    "SGD parameters must be "
                    "Tensor or Value objects"
                )


class Adam:
    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.parameters = list(parameters)

        if lr <= 0:
            raise ValueError(
                "learning rate must be positive"
            )

        if not 0.0 <= beta1 < 1.0:
            raise ValueError(
                "beta1 must satisfy 0 <= beta1 < 1"
            )

        if not 0.0 <= beta2 < 1.0:
            raise ValueError(
                "beta2 must satisfy 0 <= beta2 < 1"
            )

        if eps <= 0:
            raise ValueError(
                "eps must be positive"
            )

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.step_count = 0

        self.first_moments = []
        self.second_moments = []

        for parameter in self.parameters:
            if isinstance(parameter, Tensor):
                self.first_moments.append(
                    [0.0] * parameter.numel
                )

                self.second_moments.append(
                    [0.0] * parameter.numel
                )

            elif isinstance(parameter, Value):
                self.first_moments.append(
                    [0.0]
                )

                self.second_moments.append(
                    [0.0]
                )

            else:
                raise TypeError(
                    "Adam parameters must be "
                    "Tensor or Value objects"
                )

    def zero_grad(self):
        for parameter in self.parameters:
            if isinstance(parameter, Tensor):
                parameter.zero_grad()

            elif isinstance(parameter, Value):
                parameter.grad = 0.0

    def step(self):
        self.step_count += 1

        for parameter_index, parameter in enumerate(
            self.parameters
        ):
            first = self.first_moments[
                parameter_index
            ]

            second = self.second_moments[
                parameter_index
            ]

            if isinstance(parameter, Tensor):
                if not parameter.requires_grad:
                    continue

                if parameter.grad is None:
                    continue

                gradients = parameter.grad
                numel = parameter.numel

            elif isinstance(parameter, Value):
                gradients = [
                    parameter.grad
                ]

                numel = 1

            else:
                raise TypeError(
                    "Adam parameters must be "
                    "Tensor or Value objects"
                )

            for index in range(numel):
                gradient = gradients[index]

                first[index] = (
                    self.beta1 * first[index]
                    + (1.0 - self.beta1)
                    * gradient
                )

                second[index] = (
                    self.beta2 * second[index]
                    + (1.0 - self.beta2)
                    * gradient
                    * gradient
                )

                corrected_first = (
                    first[index]
                    / (
                        1.0
                        - self.beta1
                        ** self.step_count
                    )
                )

                corrected_second = (
                    second[index]
                    / (
                        1.0
                        - self.beta2
                        ** self.step_count
                    )
                )

                update = (
                    self.lr
                    * corrected_first
                    / (
                        math.sqrt(
                            corrected_second
                        )
                        + self.eps
                    )
                )

                if isinstance(parameter, Tensor):
                    parameter.data[index] -= update

                else:
                    parameter.data -= update


__all__ = [
    "SGD",
    "Adam",
]