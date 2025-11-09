import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask]
y = truth["ResultMeasureValue"][mask]
X = X.drop(["CharacteristicName", "id"], axis=1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

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
y = (y > 10).astype(int)

# Train/val/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosted Tree
model = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6, n_jobs=-1)
#model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=127, n_jobs=-1)
#model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=16, loss_function="MAE", eval_metric="MAE", random_seed=42, verbose=0, task_type="CPU")

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)]
)

# Training Set Evaluation
y_train_pred = model.predict(X_train)

print("\n=== Train Set Evaluation ===")
print(classification_report(y_train, y_train_pred))

# Validation Set Evaluation
y_val_pred = model.predict(X_val)

print("\n=== Val Set Evaluation ===")
print(classification_report(y_val, y_val_pred))

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_val_pred, alpha=0.5, label="Predictions")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         'r--', label="x = y (Perfect Prediction)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual (Validation Set)")
plt.legend()
plt.grid(True)
plt.show()
# Charts
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

train_cm = confusion_matrix(y_train, y_train_pred)
val_cm = confusion_matrix(y_val, y_val_pred)
y_val_proba = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
avg_precision = average_precision_score(y_val, y_val_proba)

sns.heatmap(train_cm, annot=True, cbar=True, fmt="d", cmap="Blues", xticklabels=["Safe", "Not Safe"], yticklabels=["Safe", "Not Safe"], ax=ax[0, 0])
ax[0, 0].set_xlabel("Predicted Label")
ax[0, 0].set_ylabel("Actual Label")
ax[0, 0].set_title("Training Confusion Matrix")

sns.heatmap(val_cm, annot=True, cbar=True, fmt="d", cmap="Blues", xticklabels=["Safe", "Not Safe"], yticklabels=["Safe", "Not Safe"], ax=ax[0, 1])
ax[0, 1].set_xlabel("Predicted Label")
ax[0, 1].set_ylabel("Actual Label")
ax[0, 1].set_title("Validation Confusion Matrix")

ax[1, 0].plot(recall, precision, label=f"Average precision: {avg_precision:.2f}")
ax[1, 0].set_xlabel("Recall")
ax[1, 0].set_ylabel("Precision")
ax[1, 0].set_title("Precision-Recall Curve")

plt.tight_layout()
plt.show()