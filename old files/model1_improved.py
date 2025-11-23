import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset

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

# ===== Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Finished splitting into train and test")

# ===== Convert to Tensors =====
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) #CHECK
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Split train into train/validation (80/20)
val_size = int(0.2 * len(X_train_tensor))
train_size = len(X_train_tensor) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    TensorDataset(X_train_tensor, y_train_tensor), [train_size, val_size]
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

print("Finished preprocessing, converting to tensors, and creating dataloaders")

# ===== Model Definition =====
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer_2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.layer_1(x))))
        x = self.dropout(F.relu(self.bn2(self.layer_2(x))))
        return self.output_layer(x)

input_size = X_train.shape[1]
model = BinaryClassifier(input_size)

# ===== Class-weighted Loss =====
pos_weight_value = (len(y_train) - y_train.sum()) / y_train.sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))

# ===== Optimizer & Scheduler =====
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("Finish creating model")

# ===== Training Loop =====
num_epochs = 100
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # ---- Train Accuracy ----
    model.eval()
    train_preds, train_labels_all = [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_preds.extend(preds.numpy())
            train_labels_all.extend(labels.numpy())
    train_acc = (np.array(train_preds) == np.array(train_labels_all)).mean()

    # ---- Validation ----
    val_loss = 0
    val_preds, val_labels_all = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            val_preds.extend(preds.numpy())
            val_labels_all.extend(labels.numpy())
    avg_val_loss = val_loss / len(val_loader)
    val_acc = (np.array(val_preds) == np.array(val_labels_all)).mean()

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

print("Finished training")

# ===== Threshold Tuning on Validation Set =====
model.eval()
val_logits, val_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        logits = model(inputs)
        val_logits.extend(torch.sigmoid(logits).numpy())
        val_labels.extend(labels.numpy())

val_logits = np.array(val_logits).flatten()
val_labels = np.array(val_labels).flatten()

thresholds = np.linspace(0, 1, 101)
best_threshold, best_bal_acc = 0.5, 0
for t in thresholds:
    preds = (val_logits >= t).astype(int)
    bal_acc = balanced_accuracy_score(val_labels, preds)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        best_threshold = t

print(f"Best threshold based on Balanced Accuracy: {best_threshold:.2f} (Balanced Acc={best_bal_acc:.4f})")

# ===== Final Evaluation on Test Set =====
test_logits, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        logits = model(inputs)
        test_logits.extend(torch.sigmoid(logits).numpy())
        test_labels.extend(labels.numpy())

test_logits = np.array(test_logits).flatten()
test_labels = np.array(test_labels).flatten()
test_preds = (test_logits >= best_threshold).astype(int)

# ===== Metrics =====
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
prevalence = (tp + fn) / (tn + fp + tp + fn)
bal_acc = balanced_accuracy_score(test_labels, test_preds)
cohen_kappa = cohen_kappa_score(test_labels, test_preds)
auc = roc_auc_score(test_labels, test_logits)

print("\n===== Test Set Metrics =====")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision (PPV): {ppv:.4f}")
print(f"NPV: {npv:.4f}")
print(f"Prevalence: {prevalence:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Cohen's Kappa: {cohen_kappa:.4f}")
print(f"ROC AUC: {auc:.4f}")

# ===== Plot Training Curves =====
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Time")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.show()

# ===== SHAP IMPORT =====
import shap

# ===== Prepare SHAP Background =====
# Use a small subset of training data for background (for speed)
background = X_train_tensor[:100]  # adjust if your training set is small
X_test_sample = X_test_tensor[:100]  # subset of test set for SHAP

# ===== Create SHAP Explainer =====
explainer = shap.GradientExplainer(model, background)

# ===== Compute SHAP Values =====
shap_values = explainer.shap_values(X_test_sample)

# ===== Convert tensors to numpy for plotting =====
X_test_sample_np = X_test_sample.numpy()

# ===== Plot SHAP Summary =====
# Bar plot: mean absolute SHAP values per feature
shap.summary_plot(shap_values, X_test_sample_np, feature_names=df_x.columns)

# Optional: Beeswarm plot (more detailed view of feature impact)
shap.summary_plot(shap_values, X_test_sample_np, feature_names=df_x.columns, plot_type="dot")