import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
df = pd.read_csv("(Cleaned) WQP + MRDS + gNATSGO.csv")
truth = pd.read_csv("Ground Truth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask].reset_index(drop=True)
y = truth[mask].reset_index(drop=True)
X = X.drop(["CharacteristicName", "id"], axis=1)

outlier_mask = y.values.flatten() <= 1000
X = X[outlier_mask].reset_index(drop=True)
y = y[outlier_mask].reset_index(drop=True)

mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.to_numpy().flatten()
y = y[mask]
X = X[mask]

feature_names = X.columns.tolist()
X = X.values.astype(np.float32)
X = RobustScaler().fit_transform(X)
y = y.values.astype(np.float32).reshape(-1, 1)
y = np.log1p(y)

print(X.shape)
print(y.shape)

y_val_list = []
val_predictions = []

for i in range(0, len(X), 20000):
    print(f"=== Starting Iteration {(i / 20000) + 1} ===")
    X_chunk = X[i:i+20000]
    y_chunk = y[i:i+20000]

    X_train, X_val, y_train, y_val = train_test_split(X_chunk, y_chunk, test_size=0.2)

    print("Intializing Model")
    model = TabPFNRegressor(device=device, ignore_pretraining_limits=True, random_state=42)
    print("Fitting Model")
    model.fit(X_train, y_train)
    print("Making Predictions")

    """train_predictions = np.expm1(model.predict(X_train))
    y_train.append(np.expm1(y_train))
    print("HI")"""
    val_predictions.extend(np.expm1(model.predict(X_val)).flatten())
    print("Finished Predictions")
    y_val_list.extend(np.expm1(y_val).flatten())

    """print("\n=== Train Set Evaluation ===")
    print(f"Train MAE:  {mean_absolute_error(y_train, train_predictions):.4f} µg/L")
    print(f"Train MSE:  {mean_squared_error(y_train, train_predictions):.4f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_predictions)):.4f} µg/L")
    print(f"Train R²:   {r2_score(y_train, train_predictions):.4f}")"""

    print("\n=== Validation Set Evaluation ===")
    print(f"Train MAE:  {mean_absolute_error(y_val_list, val_predictions):.4f} µg/L")
    print(f"Train MSE:  {mean_squared_error(y_val_list, val_predictions):.4f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_val_list, val_predictions)):.4f} µg/L")
    print(f"Train R²:   {r2_score(y_val_list, val_predictions):.4f}")

    """i = 3
    sample_X, sample_y = X_val[i]
    sample_X = sample_X.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y_pred_log = model(sample_X)
        y_pred = torch.expm1(y_pred_log).item()
        y_true = torch.expm1(sample_y).item()
    print(f"Prediction for item {i}: {y_pred:.4f} µg/L")
    print(f"True value for item {i}: {y_true:.4f} µg/L")"""