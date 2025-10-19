import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay,
    roc_curve
)

# ===== Load data =====
df_x = pd.read_csv("test2.csv")
df_y = pd.read_csv("testTruth.csv")

# Target variable: arsenic above WHO threshold
y = (df_y['ResultMeasureValue'] > 10).astype(int)
#X = df_x
X = df_x.drop(columns=["id", "ResultSampleFractionText_emb_0", "ResultSampleFractionText_emb_1",
                       "StateCode_emb_0", "StateCode_emb_1", "StateCode_emb_2",
                       "StateCode_emb_3", "StateCode_emb_4", "CountyCode_emb_0",
                       "CountyCode_emb_1", "CountyCode_emb_2", "CountyCode_emb_3",
                       "CountyCode_emb_4", "CountyCode_emb_5", "CountyCode_emb_6",
                       "CountyCode_emb_7", "HUCEightDigitCode_emb_0", "HUCEightDigitCode_emb_1",
                       "HUCEightDigitCode_emb_2", "HUCEightDigitCode_emb_3",
                       "HUCEightDigitCode_emb_4", "HUCEightDigitCode_emb_5",
                       "HUCEightDigitCode_emb_6", "HUCEightDigitCode_emb_7",
                       "HUCEightDigitCode_emb_8", "HUCEightDigitCode_emb_9"])

# ===== Identify numeric & categorical columns =====
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

# ===== Preprocessing =====
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ],
    remainder='drop',
    sparse_threshold=1.0
)

# ===== Train-test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Finished splitting into train and test")

# ===== Model definition =====
gbc = GradientBoostingClassifier(
    loss='log_loss', 
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)

# Combine preprocessing and model into a single pipeline
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", gbc)
])

# ===== Train =====
print("Training model...")
model.fit(X_train, y_train)
print("Finished training")

# ===== Predict =====
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ===== Evaluate basic metrics =====
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ===== Threshold Optimization (Youden’s J) =====
fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
j_scores = tpr - fpr
best_t = thresholds_roc[np.argmax(j_scores)]
print(f"Best threshold (Youden’s J): {best_t:.4f}")

# ===== Apply threshold =====
test_preds = (y_prob >= best_t).astype(int)

# ===== Confusion matrix =====
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

# ===== Detailed metrics =====
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / TPR
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # precision / PPV
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

# ===== Top Feature Importances (optional) =====
try:
    # Get feature names from the preprocessing step
    ohe = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_features = np.concatenate([numeric_cols, cat_feature_names])

    importances = model.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(order)), importances[order])
    plt.yticks(range(len(order)), all_features[order])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importances (Gain)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not plot feature importances: {e}")