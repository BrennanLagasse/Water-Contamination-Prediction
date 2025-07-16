import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

X = pd.read_csv("Testing Data.csv")
y = pd.read_csv("Testing Truth.csv")
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
print(torch.isnan(X_tensor).any(), torch.isnan(y_tensor).any())
print(torch.isinf(X_tensor).any(), torch.isinf(y_tensor).any())

class WaterDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = WaterDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

num_epochs = 100

for epoch in range(num_epochs):
    for X_batch, y_batch in loader:  # your DataLoader from earlier
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")