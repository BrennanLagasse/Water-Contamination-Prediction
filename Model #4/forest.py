import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Load data
df = pd.read_csv("WQP + MRDS.csv")
truth = pd.read_csv("Ground Truth.csv")

# Preprocess categorical column
df["ResultSampleFractionText"] = df["ResultSampleFractionText"].fillna("Missing")

# Select only Arsenic
mask = df["CharacteristicName"] == "Arsenic"
X_df = df[mask].copy()
y = truth[mask].copy()

# Embed ResultSampleFractionText
fraction_vocab = {v: i for i, v in enumerate(X_df["ResultSampleFractionText"].dropna().unique())}
X_df["fraction_idx"] = X_df["ResultSampleFractionText"].map(fraction_vocab)
fraction_emb_dim = 2
fraction_embedding = nn.Embedding(len(fraction_vocab), fraction_emb_dim)
fraction_tensor = torch.tensor(X_df["fraction_idx"].fillna(0).astype(int).values)

with torch.no_grad():
    fraction_vectors = fraction_embedding(fraction_tensor)

fraction_emb_df = pd.DataFrame(
    fraction_vectors.numpy(), 
    columns=[f"fraction_emb_{i}" for i in range(fraction_vectors.shape[1])]
)

X_df = pd.concat([X_df.reset_index(drop=True), fraction_emb_df], axis=1)

# Drop unused columns
X_df = X_df.drop(["CharacteristicName", "ResultSampleFractionText"], axis=1)

# Normalize input
X = X_df.values.astype(np.float32)
X = StandardScaler().fit_transform(X)

# Clean and log-transform target
y = y.values.astype(np.float32).reshape(-1, 1)
mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.flatten()
X = X[mask]
y = y[mask]
y_log = np.log1p(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.ravel())

# Predict
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Inverse log transform
y_pred_train = np.expm1(y_pred_train)
y_pred_test = np.expm1(y_pred_test)
y_train = np.expm1(y_train)
y_test = np.expm1(y_test)

# Evaluation function
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

# Report
print_metrics(y_train, y_pred_train, "Train Set Evaluation")
print_metrics(y_test, y_pred_test, "Test Set Evaluation")

# Plot
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([0, max(y_test.flatten())], [0, max(y_test.flatten())], color='red', linestyle='--')
plt.xlabel("True Arsenic (µg/L)")
plt.ylabel("Predicted Arsenic (µg/L)")
plt.title("Random Forest Prediction vs Truth")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()
