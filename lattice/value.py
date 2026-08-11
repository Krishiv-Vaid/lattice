import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op

        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))

        out = Value(
            self.data ** exponent,
            (self,),
            f"**{exponent}"
        )

        def _backward():
            self.grad += (
                exponent
                * (self.data ** (exponent - 1))
                * out.grad
            )

        out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(
            self.data if self.data > 0 else 0.0,
            (self,),
            "ReLU"
        )

        def _backward():
            self.grad += (
                1.0 if self.data > 0 else 0.0
            ) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(
            math.exp(self.data),
            (self,),
            "exp"
        )

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()