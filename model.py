import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Load Cleaned Data
df = pd.read_csv("WQP + SSURGO.csv")  # Replace with your file
truth = pd.read_csv("Ground Truth.csv")


# Normalizing Features
X = df.values.astype(np.float32)
y = truth.values.astype(np.float32).reshape(-1, 1)
y = np.log1p(y)
print("X mean (after norm):", X.mean(axis=0))
print("X std (after norm):", X.std(axis=0))

plt.hist(y, bins=100)
plt.title("Arsenic (µg/L) Distribution")
plt.xlabel("µg/L")
plt.ylabel("Count")
plt.yscale("log")  # helps if long tail
plt.show()

print(np.isnan(X).sum(), np.isinf(X).sum())
print(np.isnan(y).sum(), np.isinf(y).sum())

# Pytorch Dataset
class SoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SoilDataset(X, y)

# Split train / val / test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Defining Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1])

# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(300):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}")

# Validation
model.eval()
val_loss = 0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        output = model(X_batch)
        val_loss += criterion(output, y_batch).item()

        # If you want to SEE actual predictions (optional):
        predicted_arsenic = np.expm1(output.numpy())  # ← this shows real µg/L
        true_arsenic = np.expm1(y_batch.numpy())      # ← real targets
print(f"Validation Loss (log space): {val_loss / len(val_loader):.4f}")