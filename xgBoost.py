import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

# Load Data
df = pd.read_csv("(Cleaned) WQP + MRDS + gNATSGO.csv")
truth = pd.read_csv("Ground Truth.csv")

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

# Normalize features
X = RobustScaler().fit_transform(X)

# Filter invalid targets
mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.flatten()
X = X[mask]
y = y[mask]

# Log-transform target
y = np.log1p(y)

# Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Random Forest
model = xgb.XGBRegressor(
    objective="reg:absoluteerror",   # or try "reg:squarederror" if MAE stalls
    eval_metric="mae",
    n_estimators=4000,               # large; rely on early stopping
    learning_rate=0.03,              # smaller lr
    max_depth=6,                     # 3–8 is the sweet spot
    min_child_weight=10,             # prevents tiny, overfit leaves
    subsample=0.8,                   # like RF’s bagging
    colsample_bytree=0.8,            # feature sampling
    reg_alpha=0.1,                   # L1
    reg_lambda=5,                    # L2
    gamma=0,                         # you can try 1–5 if still overfitting
    tree_method="hist",              # or "gpu_hist" if on GPU
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Evaluate on validation set
y_val_pred_log = model.predict(X_val)
y_val_pred = np.expm1(y_val_pred_log)
y_val_true = np.expm1(y_val)

val_mae = mean_absolute_error(y_val_true, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
val_r2 = r2_score(y_val_true, y_val_pred)

print("\n=== Validation Set Evaluation ===")
print(f"MAE:  {val_mae:.4f} µg/L")
print(f"RMSE: {val_rmse:.4f} µg/L")
print(f"R²:   {val_r2:.4f}")

# Evaluate on test set
y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test_true = np.expm1(y_test)

mae = mean_absolute_error(y_test_true, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2 = r2_score(y_test_true, y_test_pred)

print("\n=== Final Test Set Evaluation ===")
print(f"Test MAE:  {mae:.4f} µg/L")
print(f"Test RMSE: {rmse:.4f} µg/L")
print(f"Test R²:   {r2:.4f}")

# Predict a single sample
i = 3
sample_X = X[i].reshape(1, -1)
sample_y = y[i]

y_pred_log = model.predict(sample_X)
y_pred = np.expm1(y_pred_log[0])
y_true = np.expm1(sample_y[0])

print(f"\nPrediction for item {i}: {y_pred:.4f} µg/L")
print(f"True value for item {i}: {y_true:.4f} µg/L")

# Plot predicted vs true values
plt.scatter(y_test_true, y_test_pred, alpha=0.3)
plt.plot([0, max(y_test_true.flatten())], [0, max(y_test_true.flatten())], color='red', linestyle='--')
plt.xlabel("True Arsenic (µg/L)")
plt.ylabel("Predicted Arsenic (µg/L)")
plt.title("Prediction vs Truth")
plt.xscale("log")
plt.yscale("log")
plt.grid(True)
plt.show()

# SHAP
print("HI")
import shapTest

# Create SHAP explainer and compute SHAP values
print("hi")
explainer = shapTest.TreeExplainer(model, X_train)
print("HI")
shap_values = explainer.shap_values(X_train)
print("HI")
shapTest.summary_plot(shap_values, X_train, feature_names=feature_names)
print("HI")
