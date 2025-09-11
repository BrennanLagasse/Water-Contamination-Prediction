import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ======================
# Device
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Load & Preprocess
# ======================
df = pd.read_csv("(Cleaned) WQP + MRDS + gNATSGO.csv")
truth = pd.read_csv("Ground Truth.csv")

mask = df["CharacteristicName"] == "Arsenic"
X = df[mask].reset_index(drop=True)
y = truth[mask].reset_index(drop=True)
X = X.drop(["CharacteristicName", "id"], axis=1)

outlier_mask = y.values.flatten() <= 1000
X = X[outlier_mask].reset_index(drop=True)
y = y[outlier_mask].reset_index(drop=True)

mask = (y > -1) & (~np.isnan(y)) & (~np.isinf(y))
mask = mask.to_numpy().flatten()
y = y[mask]
X = X[mask]

print("Median y:", y.median())

feature_names = X.columns.tolist()
X_mat = X.values.astype(np.float32)
X_mat = RobustScaler().fit_transform(X_mat)
y_mat = y.values.astype(np.float32).reshape(-1, 1)

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

# PyG Data
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
# PyTorch Geometric — stronger GCN: residuals, norm, dropout, Jumping Knowledge, edge dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch_geometric.nn.norm import PairNorm
from torch_geometric.utils import dropout_edge

def make_act(name: str):
    name = name.lower()
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(name, nn.ReLU)()

def make_norm(name: str, dim: int):
    name = name.lower()
    if name == "batch": return nn.BatchNorm1d(dim)
    if name == "layer": return nn.LayerNorm(dim)
    if name == "pair":  return PairNorm()
    return nn.Identity()

class BetterGCN(nn.Module):
    """
    Improvements over vanilla GCN:
      - Configurable depth
      - Residual connections
      - Batch/Layer/PairNorm
      - Dropout (features) + edge dropout
      - Jumping Knowledge (last/max/cat)
      - 2-layer MLP head
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        edge_dropout_p: float = 0.1,
        activation: str = "relu",
        norm: str = "batch",            # "batch" | "layer" | "pair" | "none"
        residual: bool = True,
        jk: str = "last",               # "last" | "max" | "cat"
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        self.dropout = dropout
        self.edge_dropout_p = edge_dropout_p
        self.act = make_act(activation)
        self.norm_name = norm.lower()
        self.residual = residual
        self.jk_mode = jk

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(make_norm(norm, hidden_channels))

        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(make_norm(norm, hidden_channels))

        # prepare JK
        self.jk = None if jk == "last" else JumpingKnowledge(mode=jk)

        # head
        head_in = hidden_channels if jk != "cat" else hidden_channels * (num_layers - 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            make_act(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index):
        if self.training and self.edge_dropout_p > 0:
            ei, _ = dropout_edge(edge_index, p=self.edge_dropout_p, force_undirected=False, training=True)
        else:
            ei = edge_index

        hs = []
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = x
            x = conv(x, ei)
            x = norm(x) if self.norm_name != "pair" else norm(x)  # PairNorm is also fine here
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual and i > 0:
                x = x + h_prev
            hs.append(x)

        x = hs[-1] if self.jk is None else self.jk(hs[1:])  # skip pre-MLP input if using "cat"/"max"
        return self.head(x)

# usage:
# model = BetterGCN(in_channels=data.num_features, hidden_channels=128, out_channels=num_classes,
#                   num_layers=5, dropout=0.5, edge_dropout_p=0.1, activation="gelu",
#                   norm="batch", residual=True, jk="max").to(device)

# Optional: GCNII (deep GCN with initial residual + identity mapping) — great for many layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv

class GCNII(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 16,
        dropout: float = 0.5,
        alpha: float = 0.1,   # initial residual weight
        theta: float = 0.5,   # identity mapping strength
    ):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.layers = nn.ModuleList()

        # linear embed -> hidden
        self.in_lin = nn.Linear(in_channels, hidden_channels)
        # stack GCN2Conv (uses initial x0 at every layer)
        for _ in range(num_layers):
            self.layers.append(GCN2Conv(channels=hidden_channels, alpha=alpha, theta=theta, layer=_ + 1))
        # head
        self.out_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x0 = F.dropout(x, p=self.dropout, training=self.training)
        x0 = F.relu(self.in_lin(x0))
        x  = x0
        for conv in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_lin(x)
        
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.lin(x)
        return x

model = BetterGCN(in_channels=data.num_features, hidden_channels=128, out_channels=1, num_layers=5, dropout=0.3, edge_dropout_p=0.1, activation="gelu", norm="batch", residual=True, jk="max").to(device)


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
y_pred_train = pred_train_log
y_true_train = true_train_log
y_pred_test = pred_test_log
y_true_test = true_test_log

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

# ======================
# Single Sample Prediction
# ======================
i = 3
model.eval()
with torch.no_grad():
    y_pred_log = model(data.x, data.edge_index)[i]
    y_pred = torch.expm1(y_pred_log).item()
    y_true = torch.expm1(data.y[i]).item()

print(f"\nPrediction for item {i}: {y_pred:.4f} µg/L")
print(f"True value for item {i}: {y_true:.4f} µg/L")
