import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from graph_models import GCN, GCNII, GAT, GraphSAGE, GIN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
df = pd.read_csv("final_dataset.csv")
truth = pd.read_csv("final_dataset_truth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask]
y = truth["ResultMeasureValue"][mask]
X = X.drop(["CharacteristicName", "id"], axis=1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# & (y <= 100)
mask = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Median y:", y.median())

# Log-transform target
y = np.log1p(y)

# Building the Graph
coords = X[["ActivityLocation/LatitudeMeasure",
            "ActivityLocation/LongitudeMeasure"]].values

X = X.values.astype(np.float32)
X = StandardScaler().fit_transform(X)
y = y.values.astype(np.float32).reshape(-1, 1)

k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
distances, indices = nbrs.kneighbors(coords)

edge_index = []
for i, neighbors in enumerate(indices):
    for j in neighbors:
        if i != j:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# PyG Data
x = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=y).to(device)

# Train/test masks
num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)
train_size = int(0.7 * num_nodes)
val_size = int(0.15 * num_nodes)
train_mask = perm[:train_size]
val_mask = perm[train_size:train_size+val_size]
test_mask = perm[train_size+val_size:]

# Defining the Model
#model = GCN(in_channels=data.num_features, hidden_channels=256, out_channels=1, num_layers=5, dropout=0.2, edge_dropout_p=0.1, activation="gelu", norm="batch", residual=True, jk="max").to(device)
#model = GCNII(in_channels=data.num_features, hidden_channels=128, out_channels=1, num_layers=16, dropout=0.5, alpha=0.1, theta=0.5).to(device)
# model = GAT(data.num_features, 128, 1, num_layers=4, heads=4, dropout=0.3, jk="max").to(device) #5.08
model = GraphSAGE(data.num_features, 128, 1, num_layers=3, dropout=0.3, jk="cat").to(device) #5.06
#model = GIN(data.num_features, 128, 1, num_layers=5, dropout=0.6, jk="last").to(device)

# Train selected Model
train_loss_list = []
val_loss_list = []

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(output[val_mask], data.y[val_mask]).item()

    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}; Validation Loss = {val_loss:.4f}")

# Training + Test Set Evaluation
model.eval()
with torch.no_grad():
    y_pred_train = model(data.x, data.edge_index)[train_mask].cpu().numpy()
    y_true_train = data.y[train_mask].cpu().numpy()

    y_pred_test = model(data.x, data.edge_index)[test_mask].cpu().numpy()
    y_true_test = data.y[test_mask].cpu().numpy()

print("\n=== Train Set Evaluation ===")
print(f"Train MAE:  {mean_absolute_error(y_true_train, y_pred_train):.4f} µg/L")
print(f"Train MSE:  {mean_squared_error(y_true_train, y_pred_train):.4f}")
print(f"Train R²:   {r2_score(y_true_train, y_pred_train):.4f}")

print("\n=== Final Test Set Evaluation ===")
print(f"Test MAE:  {mean_absolute_error(y_true_test, y_pred_test):.4f} µg/L")
print(f"Test MSE:  {mean_squared_error(y_true_test, y_pred_test):.4f}")
print(f"Test R²:   {r2_score(y_true_test, y_pred_test):.4f}")

# Loss Plot
plt.plot(train_loss_list, color="orange", label="Train Loss")
plt.plot(val_loss_list, color="blue", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_true_test, y_pred_test, alpha=0.5, label="Predictions")
plt.plot([y_true_test.min(), y_true_test.max()],
         [y_true_test.min(), y_true_test.max()],
         'r--', label="x = y (Perfect Prediction)")
plt.xlabel("Actual Values (log scale)")
plt.ylabel("Predicted Values (log scale)")
plt.title("Predicted vs Actual (Test Set)")
plt.legend()
plt.grid(True)
plt.show()