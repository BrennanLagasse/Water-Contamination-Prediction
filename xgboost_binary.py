
'''
# For the Jupyter Notebook
!pip install pandas
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
!pip install scikit-learn
!pip install numpy
!pip install matplotlib
!pip install xgboost

print("Finished installation")

!pip install --upgrade xgboost
!pip install imbalanced-learn
'''

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    cohen_kappa_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler

#Load data
df = pd.read_csv("wqp_mrds_naics_merge.csv")

#Use WHO threshold 10 µg/L to create labels (1 = dangerous, 0 = safe)
y = (df["ArsenicMeasureValue"] > 10).astype(int)

# Drop measurements from features to prevent leakage
X = df.drop(columns=["ArsenicMeasureValue"])

#Look at column types and identify if numeric or categorical
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

print(f"Total features: {X.shape[1]} | numeric: {len(numeric_cols)} | categorical: {len(categorical_cols)}")

#Split data into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Finished splitting into train and test")

#Fix class imbalance with random undersampling
# For 1:1 (pos:neg), pos fraction = 0.5
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

X_train, y_train = rus.fit_resample(X_train, y_train)

print("Counts after negative sampling:")
print(pd.Series(y_train).value_counts())

#Preprocess data
#Replace missing numeric values with median
#One-hot encode categoricals to handle unknowns
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
    ],
    remainder='drop',
    sparse_threshold=1.0
)

#Fit on train, transform splits
X_train_prep = preprocess.fit_transform(X_train)
X_test_prep = preprocess.transform(X_test)

#Build feature names for plotting importances
def get_feature_names(preprocess, numeric_cols, categorical_cols):
    names = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        ohe = preprocess.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        names.extend(cat_names)
    return np.array(names)

feature_names = get_feature_names(preprocess, numeric_cols, categorical_cols)

#Split train and validation for early stopping (80/20)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_prep, y_train, test_size=0.20, random_state=42, stratify=y_train
)

#Class imbalance handling (neg/pos)
pos = (y_tr == 1).sum()
neg = (y_tr == 0).sum()
scale_pos_weight = 1.0 #float(neg) / float(max(pos, 1)), changed because random undersampling used
print(f"Training positives: {pos}, negatives: {neg}, scale_pos_weight: {scale_pos_weight:.3f}")

#Create model
xgb = XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50
)
print("Training XGBoost with early stopping...")

#Fit with early stopping
xgb.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=False,
)
best_ntrees = xgb.best_iteration
print(f"Best iteration (trees): {best_ntrees}")

#Tune threshold on validation set with balanced accuracy
val_proba = xgb.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0, 1, 101)

best_t = 0.5
best_bal_acc = -1
for t in thresholds:
    preds = (val_proba >= t).astype(int)
    bal_acc = balanced_accuracy_score(y_val, preds)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_t = t

print(f"Best threshold by Balanced Accuracy (val): {best_t:.2f} (Balanced Acc={best_bal_acc:.4f})")

#Display final metrics from test set
test_proba = xgb.predict_proba(X_test_prep)[:, 1]
test_preds = (test_proba >= best_t).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall TPR
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # precision
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
prevalence = (tp + fn) / (tn + fp + tp + fn) if (tn + fp + tp + fn) > 0 else 0.0
bal_acc = balanced_accuracy_score(y_test, test_preds)
kappa = cohen_kappa_score(y_test, test_preds)
roc_auc = roc_auc_score(y_test, test_proba)
pr_auc = average_precision_score(y_test, test_proba)

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

'''
#Save test predicitions
out = pd.DataFrame({
    "index": X_test.index,
    "y_true": y_test.values,
    "proba_dangerous": test_proba,
    "pred_dangerous_tuned": test_preds
})
out_path = "xgb_test_predictions.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved test predictions to: {out_path}")
'''

#Look at feature importance
try:
    booster = xgb.get_booster()
    #Map importance to feature names
    score = booster.get_score(importance_type="gain")
    #Align scores to the feature names
    name_to_gain = {k: v for k, v in score.items()}
    #Get number of features from current model
    n_feats = X_train_prep.shape[1]
    keys = [f"f{i}" for i in range(n_feats)]
    gains = np.array([name_to_gain.get(k, 0.0) for k in keys])
    
    #Iterate through feature names to align with gains
    if len(feature_names) != len(gains):
        # Best effort alignment
        feature_names_local = np.array([f"f{i}" for i in range(len(gains))])
    else:
        feature_names_local = feature_names

    # Take top 20
    order = np.argsort(gains)[::-1][:20]
    top_names = feature_names_local[order]
    top_vals = gains[order]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_vals)), top_vals)
    plt.yticks(range(len(top_vals)), top_names)
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not compute feature importance plot: {e}")

#ROC Curve
try:
    RocCurveDisplay.from_predictions(y_test, test_proba)
    plt.title("ROC Curve (Test)")
    plt.show()
except Exception as e:
    print(f"Could not plot ROC curve: {e}")

#Precision Recall Curve
try:
    PrecisionRecallDisplay.from_predictions(y_test, test_proba)
    plt.title("Precision-Recall Curve (Test)")
    plt.show()
except Exception as e:
    print(f"Could not plot PR curve: {e}")