import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")

# Filter only arsenic
mask = df["CharacteristicName"] == "Arsenic"
X = df[mask].reset_index(drop=True)
y = truth[mask].reset_index(drop=True)
X = X.drop("CharacteristicName", axis=1)

# Drop outliers
outlier_mask = y.values.flatten() <= 1000
X = X[outlier_mask].reset_index(drop=True)
y = y[outlier_mask].reset_index(drop=True)

# Convert to numpy arrays
feature_names = X.columns.tolist()
X = X.values.astype(np.float32)
y = y.values.astype(np.float32).reshape(-1, 1)

# Filter invalid targets
mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.flatten()
X = X[mask]
y = y[mask]

# Log-transform target
y = np.log1p(y)

# Train/val/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train.ravel())

# === Evaluate regression on training set ===
y_train_pred_log = model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)   # inverse log transform
y_train_true = np.expm1(y_train)            # inverse log transform

train_mae = mean_absolute_error(y_train_true, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
train_r2 = r2_score(y_train_true, y_train_pred)

print("\n=== Training Set Regression Metrics ===")
print(f"MAE :  {train_mae:.4f} µg/L")
print(f"RMSE: {train_rmse:.4f} µg/L")
print(f"R²  :  {train_r2:.4f}")

# === Evaluate regression on validation set ===
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val)

val_mae = mean_absolute_error(y_val_true, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
val_r2 = r2_score(y_val_true, y_val_pred)

print("\n=== Validation Set Regression Metrics ===")
print(f"MAE :  {val_mae:.4f} µg/L")
print(f"RMSE: {val_rmse:.4f} µg/L")
print(f"R²  :  {val_r2:.4f}")