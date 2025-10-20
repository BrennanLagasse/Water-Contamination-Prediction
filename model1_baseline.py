import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay,
    roc_curve
)

# ===== LOAD DATA =====
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")
df_x = df.drop(columns=["CharacteristicName", "id"])

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask].drop(["CharacteristicName", "id"], axis=1)
y = truth["ResultMeasureValue"][mask]

# Filter valid target values
mask_valid = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X = X[mask_valid].reset_index(drop=True)
y = y[mask_valid].reset_index(drop=True)

# Convert X to float and scale
X = X.values.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Binarize target for classification
THRESHOLD = 10.0
y = (y >= THRESHOLD).astype(int)  # 0 or 1

# ===== TRAIN-TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Finished splitting into train and test")

# ===== MODEL DEFINITION =====
gbc = GradientBoostingClassifier(
    loss='log_loss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# ===== TRAIN =====
print("Training model...")
gbc.fit(X_train, y_train.to_numpy())
print("Finished training")

# ===== PREDICT =====
y_prob = gbc.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ===== EVALUATE BASIC METRICS =====
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ===== THRESHOLD OPTIMIZATION (Youden's J) =====
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_t = thresholds_roc[np.argmax(j_scores)]
print(f"Best threshold (Youden’s J): {best_t:.4f}")

# Apply threshold
test_preds = (y_prob >= best_t).astype(int)

# Confusion matrix and detailed metrics
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
prevalence = (tp + fn) / (tn + fp + tp + fn)
bal_acc = balanced_accuracy_score(y_test, test_preds)
kappa = cohen_kappa_score(y_test, test_preds)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print("\n===== Test Set Metrics =====")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity (Recall / TPR): {sensitivity:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"Precision (PPV): {ppv:.4f}")
print(f"NPV: {npv:.4f}")
print(f"Prevalence: {prevalence:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC (Average Precision): {pr_auc:.4f}")

# ===== ROC Curve =====
try:
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve (Test)")
    plt.show()
except Exception as e:
    print(f"Could not plot ROC curve: {e}")

# ===== Precision-Recall Curve =====
try:
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title("Precision-Recall Curve (Test)")
    plt.show()
except Exception as e:
    print(f"Could not plot PR curve: {e}")

# ===== SHAP VALUES (robust, shows feature names) =====
try:
    # Re-create DataFrame for X_test so we have column names
    feature_names = df_x.columns.tolist()
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Prefer newer API if available, fallback to TreeExplainer
    try:
        explainer = shap.Explainer(gbc, X_train, feature_names=feature_names)
        shap_vals = explainer(X_test_df)             # returns a ShapValues object
        # For classification, shap_vals.values shape may be (n_samples, n_classes, n_features)
        # shap.plots.* accept this ShapValues object
        # Bar summary
        shap.plots.bar(shap_vals)                    # overall importance
        # Dot summary
        shap.plots.beeswarm(shap_vals)               # per-sample impact
    except Exception:
        # Older shap versions
        explainer = shap.TreeExplainer(gbc)
        raw_shap = explainer.shap_values(X_test_df)  # may be list (n_classes) or array
        # If list (binary classifier), pick class 1 (positive)
        if isinstance(raw_shap, list) and len(raw_shap) >= 2:
            shap_for_plot = raw_shap[1]
        else:
            shap_for_plot = raw_shap
        # Summary plots (pass DataFrame to preserve names)
        shap.summary_plot(shap_for_plot, X_test_df, feature_names=feature_names, plot_type="bar", show=True)
        shap.summary_plot(shap_for_plot, X_test_df, feature_names=feature_names, show=True)

except Exception as e:
    print(f"Could not compute SHAP values: {e}")