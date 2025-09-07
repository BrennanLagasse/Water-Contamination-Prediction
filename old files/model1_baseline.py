import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

# Read merged CSV
merged = pd.read_csv("wqp_mrds_naics_merge.csv")

# Target variable: arsenic above WHO threshold
y = (merged['ArsenicMeasureValue'] > 10).astype(int)  # 10 µg/L threshold
X = merged.drop(columns=['ArsenicMeasureValue'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Finished splitting into train and test")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Finished normalization")

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(
    loss='log_loss', 
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)

print("Finish creating model")

# Train
gbc.fit(X_train_scaled, y_train)

# Predict probabilities
y_prob = gbc.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("Finished training and testing")

# Optional: Feature importance plot
plt.figure(figsize=(10,6))
plt.bar(range(X.shape[1]), gbc.feature_importances_)
plt.xticks(range(X.shape[1]), X.columns, rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score, balanced_accuracy_score
import numpy as np

# ===== Calculate Additional Metrics =====
print("===== Additional Metrics =====")

# 1. Collect predictions and probabilities
y_prob = gbc.predict_proba(X_test_scaled)[:, 1]  # Probabilities for positive class
y_pred = (y_prob >= 0.5).astype(int)             # Convert to class labels

# 2. Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 3. Metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
prevalence = (tp + fn) / (tn + fp + tp + fn)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
cohen_kappa = cohen_kappa_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# 4. Print results
print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value (PPV/Precision): {ppv:.4f}")
print(f"Negative Predictive Value (NPV): {npv:.4f}")
print(f"Prevalence: {prevalence:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Cohen's Kappa: {cohen_kappa:.4f}")
print(f"AUC: {auc:.4f}")