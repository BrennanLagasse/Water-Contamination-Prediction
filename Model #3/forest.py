import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
print("Loading WQP")
df = pd.read_csv("WQP + MRDS.csv")
print("Loading Ground Truth")
truth = pd.read_csv("Ground Truth.csv")

# Normalize features
X = df.values.astype(np.float32)
X = StandardScaler().fit_transform(X)

# Prepare target
y = truth.values.astype(np.float32).reshape(-1, 1)
mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.flatten()
X = X[mask]
y = y[mask]
y = np.log1p(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.ravel())

# Predict
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Inverse log transform
y_pred_train = np.expm1(y_pred_train)
y_train = np.expm1(y_train)
y_pred_test = np.expm1(y_pred_test)
y_test = np.expm1(y_test)

# Evaluate
def print_metrics(true, pred, name):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)
    print(f"\n=== {name} ===")
    print(f"MAE:  {mae:.4f} µg/L")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f} µg/L")
    print(f"R²:   {r2:.4f}")

print_metrics(y_train, y_pred_train, "Train Set Evaluation")
print_metrics(y_test, y_pred_test, "Test Set Evaluation")

# Plotting
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([0, max(y_test.flatten())], [0, max(y_test.flatten())], color='red', linestyle='--')
plt.xlabel("True Arsenic (µg/L)")
plt.ylabel("Predicted Arsenic (µg/L)")
plt.title("Prediction vs Truth - Random Forest")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()
