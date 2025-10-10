import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from haversine import haversine, Unit
from sklearn.neighbors import BallTree

import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    GINConv,
    JumpingKnowledge,
    BatchNorm,
    LayerNorm,
    PairNorm
)
from torch_geometric.utils import dropout_edge

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
print("Using device:", device)

# ======================
# Load & Preprocess
# ======================

df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask]
y = truth["ResultMeasureValue"][mask]
X = X.drop(["CharacteristicName", "id"], axis=1)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

mask = (y >= 0) & (y <= 1000) & (~np.isnan(y)) & (~np.isinf(y))
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Median y:", y.median())

X_mat = X.values.astype(np.float32)
X_mat = StandardScaler().fit_transform(X_mat)
y_mat = y.values.astype(np.float32).reshape(-1, 1)

y_mat = np.log1p(y_mat)
# ======================
# Build Graph (10-NN)
# ======================
coords = X[["ActivityLocation/LatitudeMeasure",
            "ActivityLocation/LongitudeMeasure"]].values

k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
distances, indices = nbrs.kneighbors(coords)

edge_index = []
for i, neighbors in enumerate(indices):
    for j in neighbors:
        if i != j:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# PCA
"""
print("Starting PCA")
pca = PCA(n_components=10)   # keep 3 principal components
X_mat = pca.fit_transform(X_mat)

print("Original shape:", X.shape)
print("Transformed shape:", X_mat.shape)

# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
"""

x = torch.tensor(X_mat, dtype=torch.float)
y = torch.tensor(y_mat, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, y=y).to(device)

# Train/test masks
num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)
train_size = int(0.7 * num_nodes)
val_size = int(0.15 * num_nodes)
train_mask = perm[:train_size]
val_mask = perm[train_size:train_size+val_size]
test_mask = perm[train_size+val_size:]

# ======================
# Model
# ======================

# ======================
# Utility Builders
# ======================
def make_act(name: str):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(name.lower(), nn.ReLU)()

def make_norm(name: str, dim: int):
    name = name.lower()
    if name == "batch": return BatchNorm(dim)
    if name == "layer": return LayerNorm(dim)
    if name == "pair":  return PairNorm()
    return nn.Identity()

# ======================
# Graph Attention Network
# ======================
class BetterGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=4,
        heads=4,
        dropout=0.5,
        edge_dropout_p=0.1,
        activation="elu",
        norm="batch",
        residual=True,
        jk="last"
    ):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.edge_dropout_p = edge_dropout_p
        self.residual = residual
        self.jk_mode = jk
        self.act = make_act(activation)

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()

        # input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.norms.append(make_norm(norm, hidden_channels * heads))

        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.norms.append(make_norm(norm, hidden_channels * heads))

        # Jumping Knowledge
        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)

        # head
        head_in = hidden_channels * heads if jk != "cat" else (hidden_channels * heads) * (num_layers - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        if self.training and self.edge_dropout_p > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.edge_dropout_p, training=True)

        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        x = hs[-1] if self.jk is None else self.jk(hs[1:])
        return self.head(x)

# ======================
# GraphSAGE
# ======================
class BetterGraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.5,
        activation="relu",
        norm="batch",
        residual=True,
        jk="last"
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.act = make_act(activation)

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()

        # input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(make_norm(norm, hidden_channels))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(make_norm(norm, hidden_channels))

        # Jumping Knowledge
        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)
        self.jk_mode = jk

        # compute correct input dim for head
        if jk == "cat":
            head_in = hidden_channels * num_layers
        else:
            head_in = hidden_channels

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        if self.jk is None:
            x = hs[-1]
        else:
            x = self.jk(hs)

        return self.head(x)

# ======================
# Graph Isomorphism Network
# ======================
class BetterGIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=4,
        dropout=0.5,
        activation="relu",
        norm="batch",
        residual=True,
        jk="last"
    ):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.act = make_act(activation)

        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        nn_linear = lambda: nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels), self.act, nn.Linear(hidden_channels, hidden_channels))))
        self.norms.append(make_norm(norm, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GINConv(nn_linear()))
            self.norms.append(make_norm(norm, hidden_channels))

        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)

        head_in = hidden_channels if jk != "cat" else hidden_channels * (num_layers - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        x = hs[-1] if self.jk is None else self.jk(hs[1:])
        return self.head(x)

# ======================
# Example usage
# ======================
# model = BetterGAT(data.num_features, 128, 1, num_layers=4, heads=4, dropout=0.3, jk="max").to(device) #5.08
# model = BetterGraphSAGE(data.num_features, 128, 1, num_layers=3, dropout=0.3, jk="cat").to(device) #5.06
model = BetterGIN(data.num_features, 128, 1, num_layers=5, dropout=0.6, jk="last").to(device)

# ======================
# Training
# ======================
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loss_list, val_loss_list = [], []

for epoch in range(500):
    # train
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    train_loss = loss.item()

    # validate
    model.eval()
    with torch.no_grad():
        val_loss = criterion(output[val_mask], data.y[val_mask]).item()
    
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}; Validation Loss = {val_loss:.4f}")

# ======================
# Final Evaluation
# ======================
model.eval()
with torch.no_grad():
    pred_train_log = model(data.x, data.edge_index)[train_mask].cpu().numpy()
    true_train_log = data.y[train_mask].cpu().numpy()

    pred_test_log = model(data.x, data.edge_index)[test_mask].cpu().numpy()
    true_test_log = data.y[test_mask].cpu().numpy()

# Undo log1p transform
y_pred_train = np.expm1(pred_train_log)
y_true_train = np.expm1(true_train_log)
y_pred_test = np.expm1(pred_test_log)
y_true_test = np.expm1(true_test_log)

print("\n=== Train Set Evaluation ===")
print(f"Train MAE:  {mean_absolute_error(y_true_train, y_pred_train):.4f} µg/L")
print(f"Train MSE:  {mean_squared_error(y_true_train, y_pred_train):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_true_train, y_pred_train)):.4f} µg/L")
print(f"Train R²:   {r2_score(y_true_train, y_pred_train):.4f}")

print("\n=== Final Test Set Evaluation ===")
print(f"Test MAE:  {mean_absolute_error(y_true_test, y_pred_test):.4f} µg/L")
print(f"Test MSE:  {mean_squared_error(y_true_test, y_pred_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_true_test, y_pred_test)):.4f} µg/L")
print(f"Test R²:   {r2_score(y_true_test, y_pred_test):.4f}")

# ======================
# Loss Plot
# ======================
plt.plot(train_loss_list, color="orange", label="Train Loss")
plt.plot(val_loss_list, color="blue", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()