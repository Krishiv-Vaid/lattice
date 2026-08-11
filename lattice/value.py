class Value:
    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data)

        return out