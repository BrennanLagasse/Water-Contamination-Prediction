import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")

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

X = X.values.astype(np.float32)
y = y.values.astype(np.float32).reshape(-1, 1)

# Log-transform target
y = np.log1p(y)

# Train/val/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train.ravel())

# Training Set Evaluation
y_train_pred = model.predict(X_train)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\n=== Train Set Evaluation ===")
print(f"Train MAE:  {train_mae:.4f} µg/L")
print(f"Train MSE:  {train_mse:.4f} µg/L")
print(f"Train R²:   {train_r2:.4f}")

# Validation Set Evaluation
y_val_pred = model.predict(X_val)

val_mae = mean_absolute_error(y_val, y_val_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("\n=== Val Set Evaluation ===")
print(f"Validation MAE:  {val_mae:.4f} µg/L")
print(f"Validation MSE:  {val_mse:.4f} µg/L")
print(f"Validation R²:   {val_r2:.4f}")