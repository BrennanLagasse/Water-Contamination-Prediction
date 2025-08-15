import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("(Cleaned) WQP + MRDS + gNATSGO.csv")
truth_classification = pd.read_csv("Ground Truth (Classification).csv")

# Filter only arsenic
mask = df["CharacteristicName"] == "Arsenic"
X = df[mask].reset_index(drop=True)
y_class = truth_classification[mask].reset_index(drop=True)
X = X.drop("CharacteristicName", axis=1)

# Convert to numpy arrays
feature_names = X.columns.tolist()
X = X.values.astype(np.float32)
y_class = y_class.values.astype(np.float32).reshape(-1, 1)

# Train/val/test split
X_train, X_val, y_class_train, y_class_val = train_test_split(X, y_class, test_size=0.2, random_state=42)

y_class_train = y_class_train.flatten().astype(int)
y_class_val   = y_class_val.flatten().astype(int)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, max_depth=16, random_state=42, n_jobs=-1)
model.fit(X_train, y_class_train.ravel())

# Evaluate on training set
y_train_pred = model.predict(X_train)

print("\n=== Training Set Evaluation ===")
print(classification_report(y_class_train, y_train_pred))

cm = confusion_matrix(y_class_train, y_train_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Evaluate on validation set
y_val_pred = model.predict(X_val)

print("\n=== Validation Set Evaluation ===")
print(classification_report(y_class_val, y_val_pred))

cm = confusion_matrix(y_class_val, y_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Predict a single sample
i = 3
sample_X = X[i].reshape(1, -1)
sample_y = y_class[i]

y_pred = model.predict(sample_X)

print(f"\nPrediction for item {i}: {y_pred} µg/L")
print(f"True value for item {i}: {sample_y} µg/L")

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
