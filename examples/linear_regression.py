from lattice.nn import MSELoss, Neuron
from lattice.optim import SGD
from lattice.value import Value


X = [
    [Value(1.0)],
    [Value(2.0)],
    [Value(3.0)],
    [Value(4.0)],
]

Y = [
    2.0,
    4.0,
    6.0,
    8.0,
]


model = Neuron(
    num_inputs=1,
    nonlinearity=False
)

loss_fn = MSELoss()

optimizer = SGD(
    model.parameters(),
    lr=0.01
)


for epoch in range(1000):
    predictions = [
        model(x)
        for x in X
    ]

    loss = loss_fn(
        predictions,
        Y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Loss {loss.data:.6f} | "
            f"Weight {model.weights[0].data:.6f} | "
            f"Bias {model.bias.data:.6f}"
        )


prediction = model(
    [Value(5.0)]
)


print()
print("Training complete")
print("Learned weight:", model.weights[0].data)
print("Learned bias:", model.bias.data)
print("Prediction for 5:", prediction.data)