import random

import lattice
import lattice.nn as nn
import lattice.optim as optim


random.seed(42)


X = [
    [lattice.Value(0.0), lattice.Value(0.0)],
    [lattice.Value(0.0), lattice.Value(1.0)],
    [lattice.Value(1.0), lattice.Value(0.0)],
    [lattice.Value(1.0), lattice.Value(1.0)],
]

Y = [
    0.0,
    1.0,
    1.0,
    0.0,
]


model = nn.MLP(
    num_inputs=2,
    layer_sizes=[4, 1]
)

# Give hidden neurons a small positive bias so ReLUs
# are less likely to start permanently inactive.
for neuron in model.layers[0].neurons:
    neuron.bias.data = 0.1


loss_fn = nn.MSELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.03
)


for epoch in range(10000):
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

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Loss {loss.data:.6f}"
        )


print()
print("Training complete")
print()

for inputs, target in zip(X, Y):
    prediction = model(inputs)

    values = [
        value.data
        for value in inputs
    ]

    print(
        f"{values} -> "
        f"{prediction.data:.4f} "
        f"(target {target})"
    )