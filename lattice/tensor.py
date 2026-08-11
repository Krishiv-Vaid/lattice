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

    def _normalize_integer_index(
        self,
        index,
        dimension_size,
    ):
        if index < 0:
            index += dimension_size

        if index < 0 or index >= dimension_size:
            raise IndexError(
                "Tensor index out of range"
            )

        return index

    def _normalize_key(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) > self.ndim:
            raise IndexError(
                "Too many indices for tensor"
            )

        key = key + (
            slice(None),
        ) * (self.ndim - len(key))

        return key

    def _iter_indices(self):
        if self.ndim == 0:
            yield ()
            return

        def recurse(dimension, prefix):
            if dimension == self.ndim:
                yield tuple(prefix)
                return

            for index in range(
                self.shape[dimension]
            ):
                prefix.append(index)

                yield from recurse(
                    dimension + 1,
                    prefix
                )

                prefix.pop()

        yield from recurse(0, [])

    def _storage_index(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) != self.ndim:
            raise IndexError(
                f"Expected {self.ndim} indices, "
                f"got {len(indices)}"
            )

        storage_index = self.offset

        for index, dimension_size, stride in zip(
            indices,
            self.shape,
            self.strides,
        ):
            if not isinstance(index, int):
                raise TypeError(
                    "Tensor indices must be integers"
                )

            index = self._normalize_integer_index(
                index,
                dimension_size,
            )

            storage_index += index * stride

        return storage_index

    def __getitem__(self, key):
        key = self._normalize_key(key)

        new_shape = []
        new_strides = []

        new_offset = self.offset

        contains_slice = False

        for item, dimension_size, stride in zip(
            key,
            self.shape,
            self.strides,
        ):
            if isinstance(item, int):
                index = self._normalize_integer_index(
                    item,
                    dimension_size,
                )

                new_offset += index * stride

            elif isinstance(item, slice):
                contains_slice = True

                start, stop, step = item.indices(
                    dimension_size
                )

                if step <= 0:
                    raise ValueError(
                        "negative and zero slice steps "
                        "are not supported yet"
                    )

                length = len(
                    range(start, stop, step)
                )

                new_offset += start * stride
                new_shape.append(length)
                new_strides.append(
                    stride * step
                )

            else:
                raise TypeError(
                    "Tensor indices must be "
                    "integers or slices"
                )

        if not contains_slice:
            return self.data[new_offset]

        return Tensor._from_storage(
            data=self.data,
            shape=new_shape,
            strides=new_strides,
            offset=new_offset,
        )

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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(
            shape[0],
            (tuple, list)
        ):
            shape = tuple(shape[0])

        if not shape:
            raise ValueError(
                "reshape requires at least one dimension"
            )

        if any(
            not isinstance(dimension, int)
            for dimension in shape
        ):
            raise TypeError(
                "reshape dimensions must be integers"
            )

        if any(
            dimension < 0
            for dimension in shape
        ):
            raise ValueError(
                "negative reshape dimensions "
                "are not supported yet"
            )

        new_numel = 1

        for dimension in shape:
            new_numel *= dimension

        if new_numel != self.numel:
            raise ValueError(
                "reshape cannot change the "
                "number of elements"
            )

        if not self.is_contiguous:
            raise ValueError(
                "cannot reshape a non-contiguous tensor; "
                "call contiguous() first"
            )

        return Tensor._from_storage(
            data=self.data,
            shape=shape,
            strides=self._compute_strides(shape),
            offset=self.offset,
        )

    def contiguous(self):
        if self.is_contiguous and self.offset == 0:
            return self

        copied_data = [
            self[index]
            for index in self._iter_indices()
        ]

        return Tensor._from_storage(
            data=copied_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
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

    @property
    def is_contiguous(self):
        return (
            self.strides
            == self._compute_strides(
                self.shape
            )
        )
        
    def _elementwise_binary_op(self, other, operation):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Tensor shapes must match for "
                    "elementwise operations"
                )

            result_data = [
                operation(
                    self[index],
                    other[index],
                )
                for index in self._iter_indices()
            ]

        elif isinstance(other, (int, float)):
            result_data = [
                operation(
                    self[index],
                    float(other),
                )
                for index in self._iter_indices()
            ]

        else:
            return NotImplemented

        return Tensor._from_storage(
            data=result_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
        )

    def __add__(self, other):
        return self._elementwise_binary_op(
            other,
            lambda a, b: a + b,
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._elementwise_binary_op(
            other,
            lambda a, b: a - b,
        )

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self._elementwise_binary_op(
                other,
                lambda a, b: b - a,
            )

        return NotImplemented

    def __mul__(self, other):
        return self._elementwise_binary_op(
            other,
            lambda a, b: a * b,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._elementwise_binary_op(
            other,
            lambda a, b: a / b,
        )

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self._elementwise_binary_op(
                other,
                lambda a, b: b / a,
            )

        return NotImplemented

    def __neg__(self):
        return self * -1.0    
    
    def __repr__(self):
        return (
            f"Tensor("
            f"shape={self.shape}, "
            f"strides={self.strides}, "
            f"offset={self.offset}, "
            f"data={self.data}"
            f")"
        )