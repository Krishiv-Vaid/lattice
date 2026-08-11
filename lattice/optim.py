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


__all__ = [
    "SGD",
]