import torch
import torch.nn as nn

# A tiny dataset: 8 examples, each with 4 input features and one target value.
x = torch.randn(8, 4)
y = x[:, 0] * 0.5 + x[:, 1] * 2.5 + torch.randn(8) * 0.05

print(f'Input shape: {x.shape}, Target shape: {y.shape}')

# You can build a model-like thing without writing a custom class.
# The usual PyTorch way is to compose layers with nn.Sequential.
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.Tanh(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

output = model(x)
print("\nSequential output shape:", output.shape)

# comparing outputs with true values
print("True values:", y)
print("Predicted values:", output.squeeze())

