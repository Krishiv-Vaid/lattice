import random

from lattice.tensor import Tensor


class TensorDataset:
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError(
                "TensorDataset requires at least one Tensor"
            )

        for tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    "TensorDataset only accepts Tensor objects"
                )

            if tensor.ndim == 0:
                raise ValueError(
                    "TensorDataset tensors must have "
                    "at least one dimension"
                )

        length = tensors[0].shape[0]

        for tensor in tensors[1:]:
            if tensor.shape[0] != length:
                raise ValueError(
                    "All tensors must have the same "
                    "size in dimension 0"
                )

        self.tensors = tuple(tensors)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError(
                "TensorDataset indices must be integers"
            )

        if index < 0:
            index += self.length

        if index < 0 or index >= self.length:
            raise IndexError(
                "TensorDataset index out of range"
            )

        return tuple(
            tensor[index]
            for tensor in self.tensors
        )


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
    ):
        if not isinstance(dataset, TensorDataset):
            raise TypeError(
                "DataLoader currently expects "
                "a TensorDataset"
            )

        if not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if not isinstance(shuffle, bool):
            raise TypeError(
                "shuffle must be a boolean"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    @staticmethod
    def _extract_sample(
        tensor,
        sample_index,
    ):
        remaining_shape = tensor.shape[1:]

        if not remaining_shape:
            return tensor[sample_index]

        def build(dimension, prefix):
            if dimension == len(
                remaining_shape
            ):
                index = (
                    sample_index,
                    *prefix,
                )

                return tensor[index]

            return [
                build(
                    dimension + 1,
                    prefix + [index],
                )
                for index in range(
                    remaining_shape[dimension]
                )
            ]

        return build(0, [])

    def __iter__(self):
        indices = list(
            range(len(self.dataset))
        )

        if self.shuffle:
            random.shuffle(indices)

        for start in range(
            0,
            len(indices),
            self.batch_size,
        ):
            batch_indices = indices[
                start:
                start + self.batch_size
            ]

            batches = []

            for tensor in self.dataset.tensors:
                batch_data = [
                    self._extract_sample(
                        tensor,
                        sample_index,
                    )
                    for sample_index
                    in batch_indices
                ]

                batches.append(
                    Tensor(batch_data)
                )

            yield tuple(batches)

    def __len__(self):
        dataset_size = len(
            self.dataset
        )

        return (
            dataset_size
            + self.batch_size
            - 1
        ) // self.batch_size


__all__ = [
    "TensorDataset",
    "DataLoader",
]