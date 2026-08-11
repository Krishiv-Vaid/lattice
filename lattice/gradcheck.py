from lattice.tensor import Tensor


def numerical_gradient(
    function,
    tensor,
    epsilon=1e-6,
):
    if not isinstance(tensor, Tensor):
        raise TypeError(
            "numerical_gradient expects a Tensor"
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive"
        )

    gradients = [
        0.0
    ] * tensor.numel

    for index in range(
        tensor.numel
    ):
        original = tensor.data[index]

        tensor.data[index] = (
            original + epsilon
        )

        positive = function().data[0]

        tensor.data[index] = (
            original - epsilon
        )

        negative = function().data[0]

        tensor.data[index] = original

        gradients[index] = (
            positive - negative
        ) / (
            2.0 * epsilon
        )

    return gradients


def gradcheck(
    function,
    tensors,
    epsilon=1e-6,
    atol=1e-5,
    rtol=1e-4,
):
    tensors = list(tensors)

    if not tensors:
        raise ValueError(
            "gradcheck requires at least one Tensor"
        )

    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(
                "gradcheck only accepts Tensor objects"
            )

        if not tensor.requires_grad:
            raise ValueError(
                "gradcheck tensors must require gradients"
            )

        tensor.zero_grad()

    loss = function()

    if not isinstance(loss, Tensor):
        raise TypeError(
            "gradcheck function must return a Tensor"
        )

    if loss.numel != 1:
        raise ValueError(
            "gradcheck function must return "
            "a scalar Tensor"
        )

    loss.backward()

    analytical = [
        list(tensor.grad)
        for tensor in tensors
    ]

    numerical = [
        numerical_gradient(
            function,
            tensor,
            epsilon=epsilon,
        )
        for tensor in tensors
    ]

    for tensor_index in range(
        len(tensors)
    ):
        analytical_gradient = analytical[
            tensor_index
        ]

        numerical_gradient_values = numerical[
            tensor_index
        ]

        for element_index, (
            analytical_value,
            numerical_value,
        ) in enumerate(
            zip(
                analytical_gradient,
                numerical_gradient_values,
            )
        ):
            difference = abs(
                analytical_value
                - numerical_value
            )

            tolerance = (
                atol
                + rtol
                * abs(numerical_value)
            )

            if difference > tolerance:
                raise AssertionError(
                    f"gradcheck failed for tensor "
                    f"{tensor_index}, element "
                    f"{element_index}: "
                    f"analytical="
                    f"{analytical_value}, "
                    f"numerical="
                    f"{numerical_value}, "
                    f"difference={difference}, "
                    f"tolerance={tolerance}"
                )

    return True


__all__ = [
    "numerical_gradient",
    "gradcheck",
]