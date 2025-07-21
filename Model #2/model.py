import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load Cleaned Data
df = pd.read_csv("WQP + SSURGO.csv")
truth = pd.read_csv("Ground Truth.csv")

# Normalizing Features
X = df.values.astype(np.float32)
X = RobustScaler().fit_transform(X)
y = truth.values.astype(np.float32).reshape(-1, 1)
y = np.log1p(y)

plt.hist(y, bins=100)
plt.title("Arsenic (µg/L) Distribution")
plt.xlabel("µg/L")
plt.ylabel("Count")
plt.yscale("log")  # helps if long tail
plt.show()

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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1])

train_loss_list = []
val_loss_list = []
# Training
criterion = nn.SmoothL1Loss(beta=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(300):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            val_loss += criterion(output, y_batch).item()

            predicted_arsenic = np.expm1(output.numpy())
            true_arsenic = np.expm1(y_batch.numpy())
    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}; Validation Loss: {val_loss:.4f}")

# Computing Errors on Test Dataset
y_pred_log = []
y_true_log = []

model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        y_pred_log.append(output.numpy())
        y_true_log.append(y_batch.numpy())

y_pred_log = np.vstack(y_pred_log)
y_pred = np.expm1(y_pred_log)
y_true_log = np.vstack(y_true_log)
y_true = np.expm1(y_true_log)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\n=== Final Test Set Evaluation ===")
print(f"Test MAE:  {mae:.4f} µg/L")
print(f"Test MSE:  {mse:.4f}")
print(f"Test RMSE: {rmse:.4f} µg/L")
print(f"Test R²:   {r2:.4f}")

i = 0  # or any index you want to test
sample_X, sample_y = dataset[i]
sample_X = sample_X.unsqueeze(0)  # Add batch dimension

model.eval()
with torch.no_grad():
    y_pred_log = model(sample_X)
    y_pred = torch.expm1(y_pred_log).item()
    y_true = torch.expm1(sample_y).item()

print(f"Prediction for item {i}: {y_pred:.4f} µg/L")
print(f"True value for item {i}: {y_true:.4f} µg/L")

# Plotting Training and Validation Graphs
plt.plot(train_loss_list, color="orange", label="Train Loss")
plt.plot(val_loss_list, color="blue", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([0, max(y_true.flatten())], [0, max(y_true.flatten())], color='red', linestyle='--')
plt.xlabel("True Arsenic (µg/L)")
plt.ylabel("Predicted Arsenic (µg/L)")
plt.title("Prediction vs Truth")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()