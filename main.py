import torch
import torch.nn as nn

# Training data
X = torch.tensor([ [1.0], [2.0], [3.0], [4.0] ])
y = torch.tensor([ [2.0], [4.0], [6.0], [8.0] ])

# A tiny neural network
model = nn.Linear(1, 1)

# Measure how wrong the model is
loss_function = nn.MSELoss()

# Update the model's parameters
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train
for epoch in range(1000):
    # Make predictions
    prediction = model(X)

    # Measure how wrong the predictions are
    loss = loss_function(prediction, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate new gradients
    loss.backward()

    # Adjust weight and bias
    optimizer.step()

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(
            "Epoch:", epoch,
            "| Loss:", loss.item(),
            "| Weight:", model.weight.item(),
            "| Bias:", model.bias.item()
        )

print("Training complete")

print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    prediction = model(test_input)
    print("Prediction for 5:", prediction.item())
