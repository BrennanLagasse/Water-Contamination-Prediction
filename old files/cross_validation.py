import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
try:
    df = pd.read_csv("(Cleaned) WQP + MRDS + gNATSGO.csv")
    truth = pd.read_csv("Ground Truth.csv")
except FileNotFoundError:
    print("Error: Data files not found. Please ensure the CSVs are in the same directory.")
    exit()

# Filter for Arsenic
mask = df["CharacteristicName"] == "Arsenic"
X = df[mask].reset_index(drop=True)
y = truth[mask].reset_index(drop=True)
X = X.drop(["CharacteristicName", "id"], axis=1)

# Filter out outliers
outlier_mask = y.values.flatten() <= 1000
X = X[outlier_mask].reset_index(drop=True)
y = y[outlier_mask].reset_index(drop=True)

# Remove invalid values
mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.to_numpy().flatten()
y = y[mask]
X = X[mask]

# Process features
feature_names = X.columns.tolist()
X = X.values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
X_scaler = RobustScaler().fit(X)
X = X_scaler.transform(X)

# Process target (log-transform only)
y = y.values.astype(np.float32).reshape(-1, 1)
y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=0.0)
y = np.where(y < 0, 0, y)
y = np.log1p(y)

print("Finished getting features")

# Dataset wrapper
class SoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = SoilDataset(X, y)
num_workers = min(8, os.cpu_count() or 1)
pin_memory = (device.type == "cuda")

# Model
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

# Sklearn-compatible wrapper
class PyTorchToSKlearnWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_class, model_params, epochs=50, lr=1e-5, device='cpu'):
        self.model_class = model_class
        self.model_params = model_params
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = None
        self.criterion = nn.L1Loss()
        self.optimizer = None

    def fit(self, X, y):
        self.model = self.model_class(**self.model_params).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5) 
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X_tensor).cpu().numpy()
    
            # clamp extreme values to avoid overflow
            output = np.clip(output, -20, 20)   # expm1(20) ≈ 4.85e8
    
            return np.expm1(output)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return r2_score(np.expm1(y_true), y_pred)

print("Finished creating model")

# Cross-validation
# Wrapper so sklearn permutation_importance can call our PyTorch model
class TorchModelWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):  
        # Dummy fit for sklearn compatibility
        return self

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).numpy().flatten()

# 10-fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_scores = []
feature_importances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold}/10 ---")

    # Split (works whether X/y are numpy arrays or pandas DataFrames)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Scale *inside* each fold
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Initialize new model per fold
    model = MLP(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(100):  # keep your epoch count
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_val_tensor).numpy().flatten()

    # Safe metric calculation
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    y_val_arr = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

    r2 = r2_score(y_val_arr, preds)
    mae = mean_absolute_error(y_val_arr, preds)

    print(f"Fold {fold} R²: {r2:.4f}, MAE: {mae:.4f}")

    fold_scores.append({'R2': r2, 'MAE': mae})

    # Permutation importance
    wrapped_model = TorchModelWrapper(model)
    perm_importance = permutation_importance(
        estimator=wrapped_model,
        X=X_val_scaled,
        y=y_val_arr,
        n_repeats=10,
        random_state=42,
        scoring="r2",
        n_jobs=-1
    )
    feature_importances.append(perm_importance.importances_mean)

print("Finished 10-fold cross validation")

# Average metrics
avg_r2 = np.mean([s['R2'] for s in fold_scores])
std_r2 = np.std([s['R2'] for s in fold_scores])
avg_mae = np.mean([s['MAE'] for s in fold_scores])
std_mae = np.std([s['MAE'] for s in fold_scores])

print(f"\n--- Cross-Validation Results ---")
print(f"Average R²: {avg_r2:.4f} ± {std_r2:.4f}")
print(f"Average MAE: {avg_mae:.4f} ± {std_mae:.4f} µg/L")

# Feature importance
avg_feature_importances = np.mean(feature_importances, axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_feature_importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance Ranking ---")
print(importance_df)

# Plot
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Feature Importance Ranking (10-Fold Cross-Validation)")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--')
plt.show()

print("Finished evaluating features")

# Compute mean importance
mean_importance = importance_df['Importance'].mean()

# Filter features above average importance
important_features_df = importance_df[importance_df['Importance'] > mean_importance]

print("\n--- Features Above Average Importance ---")
print(important_features_df)

# Save just the feature names to a new CSV
important_features_df[['Feature']].to_csv("WQP_MRDS_gNATSGO.csv", index=False)

print("Saved important features to important_features.csv")