# !pip install pandas numpy matplotlib shap xgboost scikit-learn imbalanced-learn --quiet

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    cohen_kappa_score, confusion_matrix
)
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

# LOAD + CLEAN DATA
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X_df = df[mask].drop(["CharacteristicName", "id"], axis=1)
y = truth["ResultMeasureValue"][mask]

# Clean numeric entries like '[5E-1]'
def clean_entry(v):
    if isinstance(v, str):
        s = re.sub(r"[\[\]'\"\s,]", "", v)
        try:
            return float(s)
        except ValueError:
            return np.nan
    elif pd.api.types.is_number(v):
        return v
    else:
        return np.nan

X_df = X_df.applymap(clean_entry)
X_df = X_df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
X_df = X_df.fillna(X_df.median())

mask_valid = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X_df = X_df[mask_valid].reset_index(drop=True)
y = y[mask_valid].reset_index(drop=True)

# PREPROCESSING
# Keep feature names for SHAP later
original_feature_names = X_df.columns.tolist()

# Convert + scale
X = X_df.values.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Binarize target for classification
THRESHOLD = 10.0
y = (y >= THRESHOLD).astype(int)
print("Label balance:")
print(pd.Series(y, name="ResultMeasureValue").value_counts())

# Handle class imbalance
pos = (y == 1).sum()
neg = (y == 0).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# SPLIT DATA
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
print("Finished splitting into train, validation and test")

# TRAIN BASE XGBOOST MODEL
xgb = XGBClassifier(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=100
)

xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
print(f"Best iteration: {xgb.best_iteration}")

# FEATURE SELECTION
selector = SelectFromModel(xgb, threshold="median", prefit=True)
support_mask = selector.get_support()  # Get this BEFORE transforming
selected_feature_names = np.array(original_feature_names)[support_mask]  # Store the names now

X_tr_sel = selector.transform(X_train)
X_val_sel = selector.transform(X_val)
X_test_sel = selector.transform(X_test)
print(f"Reduced features from {X.shape[1]} → {X_tr_sel.shape[1]}")
print(f"Selected features: {list(selected_feature_names)[:10]}...")  # Print first 10 as verification

# RETRAIN ON SELECTED FEATURES
xgb.fit(X_tr_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)

# OPTIMIZE CLASSIFICATION THRESHOLD
val_proba = xgb.predict_proba(X_val_sel)[:, 1]
thresholds = np.linspace(0, 1, 501)
best_t, best_score = 0.5, -np.inf

for t in thresholds:
    preds = (val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    score = 0.4 * sens + 0.6 * spec
    if score > best_score:
        best_score, best_t = score, t

print(f"Optimized threshold = {best_t:.3f} (score={best_score:.4f})")

# TEST SET EVALUATION
test_proba = xgb.predict_proba(X_test_sel)[:, 1]
test_preds = (test_proba >= best_t).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
bal_acc = balanced_accuracy_score(y_test, test_preds)
kappa = cohen_kappa_score(y_test, test_preds)
roc_auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)

print("\n===== TEST METRICS =====")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Kappa: {kappa:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")

# SHAP ANALYSIS
try:
    print(f"Using {len(selected_feature_names)} selected features for SHAP")
    
    # Build labeled DataFrames for SHAP using the selected features
    X_tr_sel_df = pd.DataFrame(X_tr_sel, columns=selected_feature_names)
    X_test_sel_df = pd.DataFrame(X_test_sel, columns=selected_feature_names)
    
    # Create SHAP TreeExplainer (specific for tree-based models like XGBoost)
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_test_sel_df)
    
    print("SHAP values computed successfully.")
    
    # For binary classification, shap_values returns values for the positive class
    # Create a proper Explanation object for the new plotting API
    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        data=X_test_sel_df.values,
        feature_names=selected_feature_names
    )
    
    # Plot SHAP values
    shap.plots.bar(shap_explanation, show=True)
    shap.plots.beeswarm(shap_explanation, show=True)

except Exception as e:
    print(f"Could not compute SHAP values: {e}")
    import traceback
    traceback.print_exc()