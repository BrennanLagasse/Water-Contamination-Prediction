import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
df = pd.read_csv("final_dataset.csv")
truth = pd.read_csv("final_dataset_truth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask]
y = truth["ResultMeasureValue"][mask]
X = X.drop(["CharacteristicName", "id"], axis=1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# & (y <= 100) 
mask = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Median y:", y.median())

print(y.shape)
X = X.values.astype(np.float32)
X = StandardScaler().fit_transform(X)
y = y.values.astype(np.float32).reshape(-1, 1)

# Log-transform target
y = np.log1p(y)

# Define and load datasets
class SoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SoilDataset(X, y)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=512, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=512, num_workers=8, pin_memory=True)

# Define model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X.shape[1]).to(device)

# Train Neural Network
train_loss_list = []
val_loss_list = []

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(150):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_loss_list.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            output = model(X_batch)
            val_loss += criterion(output, y_batch).item()
    val_loss /= len(val_loader)
    val_loss_list.append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}; Validation Loss: {val_loss:.4f}")

# Training Set Evaluation
model.eval()
y_pred_log_train, y_true_log_train = [], []
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        output = model(X_batch).cpu().numpy()
        y_pred_log_train.append(output)
        y_true_log_train.append(y_batch.numpy())

y_pred_train = np.vstack(y_pred_log_train)
y_true_train = np.vstack(y_true_log_train)

train_mae = mean_absolute_error(y_true_train, y_pred_train)
train_mse = mean_squared_error(y_true_train, y_pred_train)
train_r2 = r2_score(y_true_train, y_pred_train)

print("\n=== Train Set Evaluation ===")
print(f"Train MAE: {train_mae:.4f} µg/L")
print(f"Train MSE:  {train_mse:.4f}")
print(f"Train R²:   {train_r2:.4f}")

# Test Set Evaluation
model.eval()
y_pred_log, y_true_log = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        output = model(X_batch).cpu().numpy()
        y_pred_log.append(output)
        y_true_log.append(y_batch.numpy())

y_pred_test = np.vstack(y_pred_log)
y_true_test = np.vstack(y_true_log)

train_mae = mean_absolute_error(y_true_test, y_pred_test)
train_mse = mean_squared_error(y_true_test, y_pred_test)
train_r2 = r2_score(y_true_test, y_pred_test)

print("\n=== Test Set Evaluation ===")
print(f"Test MAE:  {train_mae:.4f} µg/L")
print(f"Test MSE:  {train_mse:.4f}")
print(f"Test R²:   {train_r2:.4f}")

# Loss Plot
plt.plot(train_loss_list, color="orange", label="Train Loss")
plt.plot(val_loss_list, color="blue", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()