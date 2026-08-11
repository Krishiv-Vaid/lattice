class Tensor:
    def __init__(self, data):
        self.data = self._flatten(data)
        self.shape = self._infer_shape(data)
        self.strides = self._compute_strides(self.shape)
        self.offset = 0

    @classmethod
    def _from_storage(
        cls,
        data,
        shape,
        strides,
        offset=0,
    ):
        tensor = cls.__new__(cls)

        tensor.data = data
        tensor.shape = tuple(shape)
        tensor.strides = tuple(strides)
        tensor.offset = offset

        return tensor

    def _infer_shape(self, data):
        if not isinstance(data, list):
            return ()

        if len(data) == 0:
            return (0,)

        first_shape = self._infer_shape(data[0])

        for item in data:
            if self._infer_shape(item) != first_shape:
                raise ValueError(
                    "Tensor data must be rectangular"
                )

        return (len(data),) + first_shape

    def _flatten(self, data):
        if not isinstance(data, list):
            return [float(data)]

        flattened = []

        for item in data:
            flattened.extend(
                self._flatten(item)
            )

        return flattened

    def _compute_strides(self, shape):
        if not shape:
            return ()

        strides = [1]

        for size in reversed(shape[1:]):
            strides.insert(
                0,
                strides[0] * size
            )

        return tuple(strides)

    def _normalize_indices(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices, "
                f"got {len(indices)}"
            )

        normalized = []

        for index, dimension_size in zip(
            indices,
            self.shape
        ):
            if not isinstance(index, int):
                raise TypeError(
                    "Tensor indices must be integers"
                )

            if index < 0:
                index += dimension_size

            if index < 0 or index >= dimension_size:
                raise IndexError(
                    "Tensor index out of range"
                )

            normalized.append(index)

        return tuple(normalized)

    def _storage_index(self, indices):
        indices = self._normalize_indices(
            indices
        )

        storage_index = self.offset

        for index, stride in zip(
            indices,
            self.strides
        ):
            storage_index += index * stride

        return storage_index

    def _normalize_dimension(self, dimension):
        if not isinstance(dimension, int):
            raise TypeError(
                "Tensor dimensions must be integers"
            )

        if dimension < 0:
            dimension += self.ndim

        if dimension < 0 or dimension >= self.ndim:
            raise IndexError(
                "Tensor dimension out of range"
            )

        return dimension

    def __getitem__(self, indices):
        storage_index = self._storage_index(
            indices
        )

        return self.data[storage_index]

    def __setitem__(self, indices, value):
        storage_index = self._storage_index(
            indices
        )

        self.data[storage_index] = float(value)

    def transpose(self, dim0, dim1):
        dim0 = self._normalize_dimension(dim0)
        dim1 = self._normalize_dimension(dim1)

        shape = list(self.shape)
        strides = list(self.strides)

        shape[dim0], shape[dim1] = (
            shape[dim1],
            shape[dim0],
        )

        strides[dim0], strides[dim1] = (
            strides[dim1],
            strides[dim0],
        )

        return Tensor._from_storage(
            data=self.data,
            shape=shape,
            strides=strides,
            offset=self.offset,
        )

    @property
    def T(self):
        if self.ndim != 2:
            raise ValueError(
                ".T is currently supported only "
                "for 2D tensors"
            )

        return self.transpose(0, 1)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        if not self.shape:
            return 1

        total = 1

        for dimension in self.shape:
            total *= dimension

        return total

    def __repr__(self):
        return (
            f"Tensor("
            f"shape={self.shape}, "
            f"strides={self.strides}, "
            f"offset={self.offset}, "
            f"data={self.data}"
            f")"
        )