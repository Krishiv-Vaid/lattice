import math

class Tensor:
    def __init__(
        self,
        data,
        requires_grad=False,
        _children=(),
        _op="",
    ):
        self.data = self._flatten(data)
        self.shape = self._infer_shape(data)
        self.strides = self._compute_strides(self.shape)
        self.offset = 0

        self.requires_grad = requires_grad

        self.grad = (
            [0.0] * self.numel
            if requires_grad
            else None
        )

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @classmethod
    def _from_storage(
        cls,
        data,
        shape,
        strides,
        offset=0,
        requires_grad=False,
        _children=(),
        _op="",
    ):
        tensor = cls.__new__(cls)

        tensor.data = data
        tensor.shape = tuple(shape)
        tensor.strides = tuple(strides)
        tensor.offset = offset

        tensor.requires_grad = requires_grad

        tensor.grad = (
            [0.0] * tensor.numel
            if requires_grad
            else None
        )

        tensor._prev = set(_children)
        tensor._op = _op
        tensor._backward = lambda: None

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

    @staticmethod
    def _iter_shape_indices(shape):
        if not shape:
            yield ()
            return

        def recurse(dimension, prefix):
            if dimension == len(shape):
                yield tuple(prefix)
                return

            for index in range(shape[dimension]):
                prefix.append(index)

                yield from recurse(
                    dimension + 1,
                    prefix,
                )

                prefix.pop()

        yield from recurse(0, [])

    def _iter_indices(self):
        yield from self._iter_shape_indices(
            self.shape
        )

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

        return key + (
            slice(None),
        ) * (self.ndim - len(key))

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

    def _flat_logical_index(self, indices):
        if not self.shape:
            return 0

        flat_index = 0

        logical_strides = self._compute_strides(
            self.shape
        )

        for index, stride in zip(
            indices,
            logical_strides,
        ):
            flat_index += index * stride

        return flat_index

    @staticmethod
    def _broadcast_shape(shape_a, shape_b):
        result = []

        reversed_a = list(reversed(shape_a))
        reversed_b = list(reversed(shape_b))

        length = max(
            len(reversed_a),
            len(reversed_b),
        )

        for i in range(length):
            dim_a = (
                reversed_a[i]
                if i < len(reversed_a)
                else 1
            )

            dim_b = (
                reversed_b[i]
                if i < len(reversed_b)
                else 1
            )

            if dim_a == dim_b:
                result.append(dim_a)

            elif dim_a == 1:
                result.append(dim_b)

            elif dim_b == 1:
                result.append(dim_a)

            else:
                raise ValueError(
                    f"Shapes {shape_a} and {shape_b} "
                    "are not broadcastable"
                )

        return tuple(reversed(result))

    @staticmethod
    def _broadcast_index(
        result_index,
        original_shape,
    ):
        if not original_shape:
            return ()

        extra_dimensions = (
            len(result_index)
            - len(original_shape)
        )

        padded_shape = (
            (1,) * extra_dimensions
            + original_shape
        )

        mapped_index = []

        for index, dimension_size in zip(
            result_index,
            padded_shape,
        ):
            if dimension_size == 1:
                mapped_index.append(0)
            else:
                mapped_index.append(index)

        if extra_dimensions:
            mapped_index = mapped_index[
                extra_dimensions:
            ]

        return tuple(mapped_index)

    def _accumulate_grad(
        self,
        indices,
        value,
    ):
        if not self.requires_grad:
            return

        flat_index = self._flat_logical_index(
            indices
        )

        self.grad[flat_index] += value

    def broadcast_to(self, shape):
        shape = tuple(shape)

        if len(shape) < self.ndim:
            raise ValueError(
                "Cannot broadcast to fewer dimensions"
            )

        padded_shape = (
            (1,) * (len(shape) - self.ndim)
            + self.shape
        )

        padded_strides = (
            (0,) * (len(shape) - self.ndim)
            + self.strides
        )

        new_strides = []

        for source_dim, target_dim, stride in zip(
            padded_shape,
            shape,
            padded_strides,
        ):
            if source_dim == target_dim:
                new_strides.append(stride)

            elif source_dim == 1:
                new_strides.append(0)

            else:
                raise ValueError(
                    f"Cannot broadcast shape "
                    f"{self.shape} to {shape}"
                )

        out = Tensor._from_storage(
            data=self.data,
            shape=shape,
            strides=new_strides,
            offset=self.offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="broadcast",
        )

        if self.requires_grad:
            def _backward():
                for out_index in out._iter_indices():
                    out_flat = (
                        out._flat_logical_index(
                            out_index
                        )
                    )

                    upstream = out.grad[
                        out_flat
                    ]

                    parent_index = (
                        self._broadcast_index(
                            out_index,
                            self.shape,
                        )
                    )

                    self._accumulate_grad(
                        parent_index,
                        upstream,
                    )

            out._backward = _backward

        return out

    def __getitem__(self, key):
        key = self._normalize_key(key)

        new_shape = []
        new_strides = []
        new_offset = self.offset

        contains_slice = False

        index_mapping = []

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

                index_mapping.append(
                    ("int", index)
                )

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

                index_mapping.append(
                    (
                        "slice",
                        start,
                        step,
                    )
                )

            else:
                raise TypeError(
                    "Tensor indices must be "
                    "integers or slices"
                )

        if not contains_slice:
            return self.data[new_offset]

        out = Tensor._from_storage(
            data=self.data,
            shape=new_shape,
            strides=new_strides,
            offset=new_offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="slice",
        )

        if self.requires_grad:
            def _backward():
                for out_index in out._iter_indices():
                    out_flat = (
                        out._flat_logical_index(
                            out_index
                        )
                    )

                    upstream = out.grad[
                        out_flat
                    ]

                    parent_index = []

                    output_dimension = 0

                    for mapping in index_mapping:
                        if mapping[0] == "int":
                            parent_index.append(
                                mapping[1]
                            )

                        else:
                            start = mapping[1]
                            step = mapping[2]

                            parent_index.append(
                                start
                                + out_index[
                                    output_dimension
                                ] * step
                            )

                            output_dimension += 1

                    self._accumulate_grad(
                        tuple(parent_index),
                        upstream,
                    )

            out._backward = _backward

        return out

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

        out = Tensor._from_storage(
            data=self.data,
            shape=shape,
            strides=strides,
            offset=self.offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        if self.requires_grad:
            def _backward():
                for out_index in out._iter_indices():
                    out_flat = out._flat_logical_index(
                        out_index
                    )

                    upstream = out.grad[out_flat]

                    parent_index = list(out_index)

                    parent_index[dim0], parent_index[dim1] = (
                        parent_index[dim1],
                        parent_index[dim0],
                    )

                    self._accumulate_grad(
                        tuple(parent_index),
                        upstream,
                    )

            out._backward = _backward

        return out

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

        out = Tensor._from_storage(
            data=self.data,
            shape=shape,
            strides=self._compute_strides(shape),
            offset=self.offset,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        if self.requires_grad:
            def _backward():
                for out_index in out._iter_indices():
                    flat = out._flat_logical_index(
                        out_index
                    )

                    upstream = out.grad[flat]

                    self.grad[flat] += upstream

            out._backward = _backward

        return out

    def contiguous(self):
        if self.is_contiguous and self.offset == 0:
            return self

        copied_data = [
            self[index]
            for index in self._iter_indices()
        ]

        out = Tensor._from_storage(
            data=copied_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="contiguous",
        )

        if self.requires_grad:
            def _backward():
                for index in out._iter_indices():
                    flat = out._flat_logical_index(
                        index
                    )

                    upstream = out.grad[flat]

                    self._accumulate_grad(
                        index,
                        upstream,
                    )

            out._backward = _backward

        return out

    def _elementwise_binary_op(
        self,
        other,
        operation,
        op_name,
    ):
        if isinstance(other, Tensor):
            result_shape = self._broadcast_shape(
                self.shape,
                other.shape,
            )

            left = self.broadcast_to(
                result_shape
            )

            right = other.broadcast_to(
                result_shape
            )

            result_data = [
                operation(
                    left[index],
                    right[index],
                )
                for index in left._iter_indices()
            ]

            requires_grad = (
                self.requires_grad
                or other.requires_grad
            )

            out = Tensor._from_storage(
                data=result_data,
                shape=result_shape,
                strides=self._compute_strides(
                    result_shape
                ),
                offset=0,
                requires_grad=requires_grad,
                _children=(self, other),
                _op=op_name,
            )

            if requires_grad:
                def _backward():
                    for index in out._iter_indices():
                        out_flat = (
                            out._flat_logical_index(
                                index
                            )
                        )

                        upstream = out.grad[
                            out_flat
                        ]

                        if self.requires_grad:
                            self_index = (
                                self._broadcast_index(
                                    index,
                                    self.shape,
                                )
                            )

                            if op_name == "+":
                                local = 1.0

                            elif op_name == "*":
                                local = right[index]

                            else:
                                local = 0.0

                            self._accumulate_grad(
                                self_index,
                                local * upstream,
                            )

                        if other.requires_grad:
                            other_index = (
                                self._broadcast_index(
                                    index,
                                    other.shape,
                                )
                            )

                            if op_name == "+":
                                local = 1.0

                            elif op_name == "*":
                                local = left[index]

                            else:
                                local = 0.0

                            other._accumulate_grad(
                                other_index,
                                local * upstream,
                            )

                out._backward = _backward

            return out

        elif isinstance(other, (int, float)):
            result_data = [
                operation(
                    self[index],
                    float(other),
                )
                for index in self._iter_indices()
            ]

            out = Tensor._from_storage(
                data=result_data,
                shape=self.shape,
                strides=self._compute_strides(
                    self.shape
                ),
                offset=0,
                requires_grad=self.requires_grad,
                _children=(self,),
                _op=op_name,
            )

            if self.requires_grad:
                def _backward():
                    for index in out._iter_indices():
                        flat = (
                            out._flat_logical_index(
                                index
                            )
                        )

                        upstream = out.grad[flat]

                        if op_name == "+":
                            local = 1.0

                        elif op_name == "*":
                            local = float(other)

                        else:
                            local = 0.0

                        self._accumulate_grad(
                            index,
                            local * upstream,
                        )

                out._backward = _backward

            return out

        return NotImplemented

    def _reduce_forward(
        self,
        dim,
        operation,
        initial,
    ):
        dim = self._normalize_dimension(dim)

        result_shape = (
            self.shape[:dim]
            + self.shape[dim + 1:]
        )

        result_data = []

        for output_index in self._iter_shape_indices(
            result_shape
        ):
            total = initial

            for reduced_index in range(
                self.shape[dim]
            ):
                full_index = list(output_index)

                full_index.insert(
                    dim,
                    reduced_index,
                )

                total = operation(
                    total,
                    self[tuple(full_index)],
                )

            result_data.append(total)

        return Tensor._from_storage(
            data=result_data,
            shape=result_shape,
            strides=self._compute_strides(
                result_shape
            ),
            offset=0,
            requires_grad=False,
        )

    def backward(self):
        if self.numel != 1:
            raise ValueError(
                "backward() currently requires "
                "a scalar tensor"
            )

        if not self.requires_grad:
            raise ValueError(
                "Cannot call backward() on a tensor "
                "that does not require gradients"
            )

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        # Intermediate graph nodes must start each
        # backward pass with fresh gradients.
        #
        # Leaf tensors intentionally keep their
        # gradients so repeated backward() calls
        # accumulate into them.
        for node in topo:
            if (
                node.requires_grad
                and node._prev
            ):
                node.grad = [
                    0.0
                ] * node.numel

        # Seed dL/dL = 1.
        #
        # If the root itself is a leaf scalar,
        # accumulate just like any other leaf.
        if self._prev:
            self.grad[0] = 1.0
        else:
            self.grad[0] += 1.0

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = [0.0] * self.numel

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(
                "matmul requires another Tensor"
            )

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError(
                "matmul currently supports only 2D tensors"
            )

        rows_a, cols_a = self.shape
        rows_b, cols_b = other.shape

        if cols_a != rows_b:
            raise ValueError(
                f"Cannot multiply tensors with shapes "
                f"{self.shape} and {other.shape}"
            )

        result_data = []

        for i in range(rows_a):
            for j in range(cols_b):
                total = 0.0

                for k in range(cols_a):
                    total += (
                        self[i, k]
                        * other[k, j]
                    )

                result_data.append(total)

        result_shape = (
            rows_a,
            cols_b,
        )

        requires_grad = (
            self.requires_grad
            or other.requires_grad
        )

        out = Tensor._from_storage(
            data=result_data,
            shape=result_shape,
            strides=self._compute_strides(
                result_shape
            ),
            offset=0,
            requires_grad=requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        if requires_grad:
            def _backward():
                if self.requires_grad:
                    for i in range(rows_a):
                        for k in range(cols_a):
                            total = 0.0

                            for j in range(cols_b):
                                out_index = (
                                    i,
                                    j,
                                )

                                out_flat = (
                                    out._flat_logical_index(
                                        out_index
                                    )
                                )

                                upstream = out.grad[
                                    out_flat
                                ]

                                total += (
                                    upstream
                                    * other[k, j]
                                )

                            self._accumulate_grad(
                                (i, k),
                                total,
                            )

                if other.requires_grad:
                    for k in range(rows_b):
                        for j in range(cols_b):
                            total = 0.0

                            for i in range(rows_a):
                                out_index = (
                                    i,
                                    j,
                                )

                                out_flat = (
                                    out._flat_logical_index(
                                        out_index
                                    )
                                )

                                upstream = out.grad[
                                    out_flat
                                ]

                                total += (
                                    self[i, k]
                                    * upstream
                                )

                            other._accumulate_grad(
                                (k, j),
                                total,
                            )

            out._backward = _backward

        return out

    def sum(self, dim=None):
        if dim is None:
            total = 0.0

            for index in self._iter_indices():
                total += self[index]

            out = Tensor(
                total,
                requires_grad=self.requires_grad,
                _children=(self,),
                _op="sum",
            )

            if self.requires_grad:
                def _backward():
                    upstream = out.grad[0]

                    for index in self._iter_indices():
                        self._accumulate_grad(
                            index,
                            upstream,
                        )

                out._backward = _backward

            return out

        dim = self._normalize_dimension(dim)

        result_shape = (
            self.shape[:dim]
            + self.shape[dim + 1:]
        )

        result_data = []

        for output_index in self._iter_shape_indices(
            result_shape
        ):
            total = 0.0

            for reduced_index in range(
                self.shape[dim]
            ):
                full_index = list(output_index)

                full_index.insert(
                    dim,
                    reduced_index,
                )

                total += self[
                    tuple(full_index)
                ]

            result_data.append(total)

        out = Tensor._from_storage(
            data=result_data,
            shape=result_shape,
            strides=self._compute_strides(
                result_shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        if self.requires_grad:
            def _backward():
                for output_index in out._iter_indices():
                    out_flat = (
                        out._flat_logical_index(
                            output_index
                        )
                    )

                    upstream = out.grad[
                        out_flat
                    ]

                    for reduced_index in range(
                        self.shape[dim]
                    ):
                        full_index = list(
                            output_index
                        )

                        full_index.insert(
                            dim,
                            reduced_index,
                        )

                        self._accumulate_grad(
                            tuple(full_index),
                            upstream,
                        )

            out._backward = _backward

        return out

    def mean(self, dim=None):
        if dim is None:
            if self.numel == 0:
                raise ValueError(
                    "mean of an empty tensor is undefined"
                )

            return self.sum() / self.numel

        dim = self._normalize_dimension(dim)

        if self.shape[dim] == 0:
            raise ValueError(
                "mean of an empty dimension is undefined"
            )

        return (
            self.sum(dim=dim)
            / self.shape[dim]
        )


    def relu(self):
        result_data = [
            max(0.0, self[index])
            for index in self._iter_indices()
        ]

        out = Tensor._from_storage(
            data=result_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        if self.requires_grad:
            def _backward():
                for index in out._iter_indices():
                    flat = out._flat_logical_index(
                        index
                    )

                    upstream = out.grad[flat]

                    local = (
                        1.0
                        if self[index] > 0.0
                        else 0.0
                    )

                    self._accumulate_grad(
                        index,
                        local * upstream,
                    )

            out._backward = _backward

        return out

    def __add__(self, other):
        return self._elementwise_binary_op(
            other,
            lambda a, b: a + b,
            "+",
        )

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        return self._elementwise_binary_op(
            other,
            lambda a, b: a * b,
            "*",
        )

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)

        if isinstance(other, Tensor):
            if (
                self.requires_grad
                or other.requires_grad
            ):
                raise NotImplementedError(
                    "Tensor/Tensor division autograd "
                    "is not implemented yet"
                )

            return self._elementwise_binary_op(
                other,
                lambda a, b: a / b,
                "/",
            )

        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            if self.requires_grad:
                raise NotImplementedError(
                    "Scalar/Tensor division autograd "
                    "is not implemented yet"
                )

            return self._elementwise_binary_op(
                other,
                lambda a, b: b / a,
                "/",
            )

        return NotImplemented

    def __neg__(self):
        return self * -1.0

    def __matmul__(self, other):
        return self.matmul(other)

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

    def __repr__(self):
        return (
            f"Tensor("
            f"shape={self.shape}, "
            f"strides={self.strides}, "
            f"offset={self.offset}, "
            f"requires_grad={self.requires_grad}, "
            f"data={self.data}"
            f")"
        )
        
    def exp(self):
        result_data = [
            math.exp(self[index])
            for index in self._iter_indices()
        ]

        out = Tensor._from_storage(
            data=result_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        if self.requires_grad:
            def _backward():
                for index in out._iter_indices():
                    flat = out._flat_logical_index(
                        index
                    )

                    upstream = out.grad[flat]

                    self._accumulate_grad(
                        index,
                        out[index] * upstream,
                    )

            out._backward = _backward

        return out

    def log(self):
        for index in self._iter_indices():
            if self[index] <= 0.0:
                raise ValueError(
                    "log is only defined for "
                    "positive tensor values"
                )

        result_data = [
            math.log(self[index])
            for index in self._iter_indices()
        ]

        out = Tensor._from_storage(
            data=result_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        if self.requires_grad:
            def _backward():
                for index in out._iter_indices():
                    flat = out._flat_logical_index(
                        index
                    )

                    upstream = out.grad[flat]

                    self._accumulate_grad(
                        index,
                        upstream / self[index],
                    )

            out._backward = _backward

        return out
    
    def softmax(self, dim=-1):
        dim = self._normalize_dimension(dim)

        result_data = [
            0.0
        ] * self.numel

        outer_shape = (
            self.shape[:dim]
            + self.shape[dim + 1:]
        )

        for outer_index in self._iter_shape_indices(
            outer_shape
        ):
            values = []

            for dim_index in range(
                self.shape[dim]
            ):
                full_index = list(outer_index)
                full_index.insert(
                    dim,
                    dim_index,
                )

                values.append(
                    self[tuple(full_index)]
                )

            maximum = max(values)

            exponentials = [
                math.exp(value - maximum)
                for value in values
            ]

            denominator = sum(exponentials)

            for dim_index, exponential in enumerate(
                exponentials
            ):
                full_index = list(outer_index)
                full_index.insert(
                    dim,
                    dim_index,
                )

                flat = self._flat_logical_index(
                    tuple(full_index)
                )

                result_data[flat] = (
                    exponential / denominator
                )

        out = Tensor._from_storage(
            data=result_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="softmax",
        )

        if self.requires_grad:
            def _backward():
                for outer_index in self._iter_shape_indices(
                    outer_shape
                ):
                    dot = 0.0

                    for dim_index in range(
                        self.shape[dim]
                    ):
                        full_index = list(outer_index)
                        full_index.insert(
                            dim,
                            dim_index,
                        )

                        full_index = tuple(full_index)

                        flat = out._flat_logical_index(
                            full_index
                        )

                        dot += (
                            out.grad[flat]
                            * out[full_index]
                        )

                    for dim_index in range(
                        self.shape[dim]
                    ):
                        full_index = list(outer_index)
                        full_index.insert(
                            dim,
                            dim_index,
                        )

                        full_index = tuple(full_index)

                        flat = out._flat_logical_index(
                            full_index
                        )

                        gradient = (
                            out[full_index]
                            * (
                                out.grad[flat]
                                - dot
                            )
                        )

                        self._accumulate_grad(
                            full_index,
                            gradient,
                        )

            out._backward = _backward

        return out
    
    def log_softmax(self, dim=-1):
        dim = self._normalize_dimension(dim)

        result_data = [
            0.0
        ] * self.numel

        softmax_data = [
            0.0
        ] * self.numel

        outer_shape = (
            self.shape[:dim]
            + self.shape[dim + 1:]
        )

        for outer_index in self._iter_shape_indices(
            outer_shape
        ):
            values = []

            for dim_index in range(
                self.shape[dim]
            ):
                full_index = list(outer_index)
                full_index.insert(
                    dim,
                    dim_index,
                )

                values.append(
                    self[tuple(full_index)]
                )

            maximum = max(values)

            exponentials = [
                math.exp(value - maximum)
                for value in values
            ]

            denominator = sum(exponentials)

            log_denominator = math.log(
                denominator
            )

            for dim_index, value in enumerate(
                values
            ):
                full_index = list(outer_index)
                full_index.insert(
                    dim,
                    dim_index,
                )

                full_index = tuple(full_index)

                flat = self._flat_logical_index(
                    full_index
                )

                result_data[flat] = (
                    value
                    - maximum
                    - log_denominator
                )

                softmax_data[flat] = (
                    exponentials[dim_index]
                    / denominator
                )

        out = Tensor._from_storage(
            data=result_data,
            shape=self.shape,
            strides=self._compute_strides(
                self.shape
            ),
            offset=0,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log_softmax",
        )

        if self.requires_grad:
            def _backward():
                for outer_index in self._iter_shape_indices(
                    outer_shape
                ):
                    grad_sum = 0.0

                    for dim_index in range(
                        self.shape[dim]
                    ):
                        full_index = list(outer_index)
                        full_index.insert(
                            dim,
                            dim_index,
                        )

                        flat = out._flat_logical_index(
                            tuple(full_index)
                        )

                        grad_sum += out.grad[flat]

                    for dim_index in range(
                        self.shape[dim]
                    ):
                        full_index = list(outer_index)
                        full_index.insert(
                            dim,
                            dim_index,
                        )

                        full_index = tuple(full_index)

                        flat = out._flat_logical_index(
                            full_index
                        )

                        gradient = (
                            out.grad[flat]
                            - softmax_data[flat]
                            * grad_sum
                        )

                        self._accumulate_grad(
                            full_index,
                            gradient,
                        )

            out._backward = _backward

        return out
    
    def argmax(self, dim=None):
        if self.numel == 0:
            raise ValueError(
                "argmax is undefined for an empty tensor"
            )

        if dim is None:
            best_flat = 0
            best_value = self.data[
                self._storage_index(
                    tuple(
                        0
                        for _ in range(self.ndim)
                    )
                )
            ] if self.ndim > 0 else self.data[self.offset]

            for flat, index in enumerate(
                self._iter_indices()
            ):
                value = self[index]

                if value > best_value:
                    best_value = value
                    best_flat = flat

            return best_flat

        dim = self._normalize_dimension(dim)

        result_shape = (
            self.shape[:dim]
            + self.shape[dim + 1:]
        )

        result_data = []

        for output_index in self._iter_shape_indices(
            result_shape
        ):
            best_index = 0

            full_index = list(output_index)
            full_index.insert(
                dim,
                0,
            )

            best_value = self[
                tuple(full_index)
            ]

            for dim_index in range(
                1,
                self.shape[dim]
            ):
                full_index = list(output_index)
                full_index.insert(
                    dim,
                    dim_index,
                )

                value = self[
                    tuple(full_index)
                ]

                if value > best_value:
                    best_value = value
                    best_index = dim_index

            result_data.append(
                float(best_index)
            )

        return Tensor._from_storage(
            data=result_data,
            shape=result_shape,
            strides=self._compute_strides(
                result_shape
            ),
            offset=0,
            requires_grad=False,
        )