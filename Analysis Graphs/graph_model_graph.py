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
import shap

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Data
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")

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
model = GAT(data.num_features, 128, 1, num_layers=4, heads=4, dropout=0.3, jk="max").to(device) #5.08
# model = GraphSAGE(data.num_features, 128, 1, num_layers=3, dropout=0.3, jk="cat").to(device) #5.06
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
    y_pred_test = model(data.x, data.edge_index)[test_mask].cpu().numpy()
    y_true_test = data.y[test_mask].cpu().numpy()

print("\n=== Final Test Set Evaluation ===")
print(f"Test MAE:  {mean_absolute_error(y_true_test, y_pred_test):.4f} µg/L")
print(f"Test MSE:  {mean_squared_error(y_true_test, y_pred_test):.4f}")
print(f"Test R²:   {r2_score(y_true_test, y_pred_test):.4f}")

# Visualization
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

# Convert tensors to NumPy
y_pred_lin = np.expm1(y_pred_test.flatten())
y_true_lin = np.expm1(y_true_test.flatten())

# Compute residuals in log(x+1) space
error = np.log1p(y_pred_lin) - np.log1p(y_true_lin)

# Match test coordinates
coords_test = coords[test_mask.cpu().numpy()]
lat = coords_test[:, 0]
lon = coords_test[:, 1]

# Predicted vs Actual Scatter
plt.figure(figsize=(6,6))
plt.scatter(y_true_test, y_pred_test, alpha=0.5, label="Predictions")
plt.plot([y_true_test.min(), y_true_test.max()],
         [y_true_test.min(), y_true_test.max()],
         'r--', label="x = y (Perfect Prediction)")
plt.xlabel("Actual log(x+1) Values")
plt.ylabel("Predicted log(x+1) Values")
plt.title("Predicted vs Actual in log(x+1) scale (GraphSAGE, Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Spatial Error Map
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-125, -66, 25, 49], crs=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4)

# Normalize color scale around zero
vmin, vmax = np.min(error), np.max(error)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Scatter plot
sc = ax.scatter(
    lon, lat,
    c=error,
    cmap="RdBu_r",
    norm=norm,
    s=18,
    alpha=0.7,
    transform=ccrs.PlateCarree()
)

# Colorbar
cbar = plt.colorbar(sc, orientation='vertical', pad=0.02, shrink=0.8)
cbar.set_label("Log Residual (Predicted − Actual)", fontsize=11)
plt.title("Spatial Distribution of Arsenic Prediction Error (GraphSAGE, Test Set)", fontsize=12)
plt.tight_layout()
plt.show()