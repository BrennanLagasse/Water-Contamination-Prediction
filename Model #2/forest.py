import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

# Load Cleaned Data
df = pd.read_csv("WQP + SSURGO.csv")  # Replace with your file
truth = pd.read_csv("Ground Truth.csv")

# Normalize Target
X = df.values.astype(np.float32)
X = RobustScaler().fit_transform(X)
y = truth.values.astype(np.float32).reshape(-1)
y_log = np.log1p(y)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y_log, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validation
y_val_pred_log = model.predict(X_val)
y_val = np.expm1(y_val)
y_val_pred_log = np.expm1(y_val_pred_log)
val_loss = mean_squared_error(y_val, y_val_pred_log)
print(f"Validation Loss MSE: {val_loss:.4f}")
val_loss = mean_absolute_error(y_val, y_val_pred_log)
print(f"Validation Loss MAE: {val_loss:.4f}")
val_loss = r2_score(y_val, y_val_pred_log)
print(f"Validation Loss R2: {val_loss:.4f}")

# Optional: Convert predictions back to µg/L space and visualize
y_val_pred_real = np.expm1(y_val_pred_log)
y_val_real = np.expm1(y_val)

plt.scatter(y_val_real, y_val_pred_real, alpha=0.5)
plt.plot([y_val_real.min(), y_val_real.max()], [y_val_real.min(), y_val_real.max()], 'r--')
plt.xlabel("True Arsenic (µg/L)")
plt.ylabel("Predicted Arsenic (µg/L)")
plt.title("Random Forest Predictions vs Truth")
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()

from xgboost import XGBRegressor

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Validation
y_val_pred_xgb_log = xgb_model.predict(X_val)
xgb_val_loss = mean_squared_error(y_val, y_val_pred_xgb_log)
print(f"XGBoost Validation Loss (log space): {xgb_val_loss:.4f}")

# Optional: Convert back to real units
y_val_pred_xgb_real = np.expm1(y_val_pred_xgb_log)

# Plot comparison
plt.scatter(y_val_real, y_val_pred_xgb_real, alpha=0.5, label='XGBoost')
plt.scatter(y_val_real, y_val_pred_real, alpha=0.3, label='Random Forest')
plt.plot([y_val_real.min(), y_val_real.max()], [y_val_real.min(), y_val_real.max()], 'r--')
plt.xlabel("True Arsenic (µg/L)")
plt.ylabel("Predicted Arsenic (µg/L)")
plt.title("XGBoost vs Random Forest Predictions")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()