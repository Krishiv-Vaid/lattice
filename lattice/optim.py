class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0.0

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad