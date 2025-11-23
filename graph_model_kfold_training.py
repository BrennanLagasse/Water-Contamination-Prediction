import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data
from graph_models import GAT
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*70)
print(f"Using device: {device}")
print("="*70)

# Load Data
print("\nLoading data...")
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")
wqp_original = pd.read_csv("WQP Physical Chemical.csv", low_memory=False)
station = pd.read_csv("WQP Station Metadata.csv", low_memory=False)
fold_assignments = pd.read_csv("huc4_fold_assignments.csv")

# Filter to Arsenic only
arsenic_mask = df["CharacteristicName"] == "Arsenic"
X = df[arsenic_mask].copy()
y = truth["ResultMeasureValue"][arsenic_mask].copy()
X = X.drop(["CharacteristicName", "id"], axis=1)

print(f"Initial X shape: {X.shape}")
print(f"Initial y shape: {y.shape}")

# Extract HUC4 from original WQP data then filter valid values
print("\nExtracting HUC codes from original data...")
wqp_arsenic = wqp_original[wqp_original["CharacteristicName"] == "Arsenic"].copy()

# Merge with station data to get HUC codes
wqp_with_huc = wqp_arsenic[["MonitoringLocationIdentifier", "ActivityLocation/LatitudeMeasure", 
                             "ActivityLocation/LongitudeMeasure"]].merge(
    station[["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "HUCEightDigitCode"]],
    on="MonitoringLocationIdentifier",
    how="left"
)

# Fill missing coordinates
wqp_with_huc['ActivityLocation/LatitudeMeasure'] = wqp_with_huc['ActivityLocation/LatitudeMeasure'].fillna(
    wqp_with_huc['LatitudeMeasure'])
wqp_with_huc['ActivityLocation/LongitudeMeasure'] = wqp_with_huc['ActivityLocation/LongitudeMeasure'].fillna(
    wqp_with_huc['LongitudeMeasure'])

# Extract HUC4 and add to X then filter
X['HUC4'] = wqp_with_huc['HUCEightDigitCode'].astype(str).str[:4].astype(str)

# Filter valid values
valid_mask = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X = X[valid_mask].reset_index(drop=True)
y = y[valid_mask].reset_index(drop=True)

print(f"After filtering - X shape: {X.shape}, y shape: {y.shape}")
print(f"Median y: {y.median():.4f}")

# Merge with fold assignments
fold_assignments['HUC4'] = fold_assignments['HUC4'].astype(str)
X = X.merge(fold_assignments, on='HUC4', how='left')

# Check for missing fold assignments
missing_folds = X['fold'].isna().sum()
if missing_folds > 0:
    print(f"{missing_folds} samples have no fold assignment & will be dropped")
    valid_fold_mask = X['fold'].notna()
    X = X[valid_fold_mask].reset_index(drop=True)
    y = y[valid_fold_mask].reset_index(drop=True)

print(f"After fold assignment - X shape: {X.shape}, y shape: {y.shape}")
print(f"Fold distribution:\n{X['fold'].value_counts().sort_index()}")

# Save fold column and drop from features
fold_labels = X['fold'].values.astype(int)
X = X.drop(['fold', 'HUC4'], axis=1)

# Log-transform target
y = np.log1p(y)

# Build Graph
print("\nBuilding graph structure...")
coords = X[["ActivityLocation/LatitudeMeasure",
            "ActivityLocation/LongitudeMeasure"]].values

X_array = X.values.astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)
y_array = y.values.astype(np.float32).reshape(-1, 1)

k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(coords)
distances, indices = nbrs.kneighbors(coords)

edge_index = []
for i, neighbors in enumerate(indices):
    for j in neighbors:
        if i != j:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print(f"Graph constructed with {len(X_scaled)} nodes and {edge_index.shape[1]} edges")

x = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y_array, dtype=torch.float)

# K-Fold Cross-Validation
K_FOLDS = len(np.unique(fold_labels))
print(f"\n{'='*70}")
print(f"STARTING {K_FOLDS}-FOLD GEOGRAPHIC CROSS-VALIDATION")
print(f"{'='*70}")

# Storage for results
fold_results = []

for fold_idx in range(K_FOLDS):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/{K_FOLDS}")
    print(f"{'='*70}")
    
    # Create train/test masks based on fold assignment
    test_mask = torch.tensor(fold_labels == fold_idx, dtype=torch.bool)
    train_mask = torch.tensor(fold_labels != fold_idx, dtype=torch.bool)
    
    n_train = train_mask.sum().item()
    n_test = test_mask.sum().item()
    print(f"Train samples: {n_train:,} ({n_train/len(X_scaled)*100:.1f}%)")
    print(f"Test samples:  {n_test:,} ({n_test/len(X_scaled)*100:.1f}%)")
    
    data = Data(x=x, edge_index=edge_index, y=y_tensor).to(device)
    
    # Initialize model
    model = GAT(
        data.num_features, 
        128, 
        1, 
        num_layers=4, 
        heads=4, 
        dropout=0.3, 
        jk="max"
    ).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    best_train_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        
        # Early stopping based on training loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/500: Train Loss = {train_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(data.x, data.edge_index)[test_mask].cpu().numpy()
        y_true_test = data.y[test_mask].cpu().numpy()
    
    # Metrics in log space
    mae_log = mean_absolute_error(y_true_test, y_pred_test)
    mse_log = mean_squared_error(y_true_test, y_pred_test)
    r2_log = r2_score(y_true_test, y_pred_test)
    
    # Metrics in original space
    y_pred_orig = np.expm1(y_pred_test.flatten())
    y_true_orig = np.expm1(y_true_test.flatten())
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
    r2_orig = r2_score(y_true_orig, y_pred_orig)
    rmse_orig = np.sqrt(mse_orig)
    
    print(f"\n--- Fold {fold_idx + 1} Results ---")
    print(f"Log Space:  MAE={mae_log:.4f}, MSE={mse_log:.4f}, R²={r2_log:.4f}")
    print(f"Real Space: MAE={mae_orig:.4f} µg/L, RMSE={rmse_orig:.4f} µg/L, R²={r2_orig:.4f}")
    
    fold_results.append({
        'fold': fold_idx + 1,
        'n_train': n_train,
        'n_test': n_test,
        'mae_log': mae_log,
        'mse_log': mse_log,
        'r2_log': r2_log,
        'mae_orig': mae_orig,
        'mse_orig': mse_orig,
        'rmse_orig': rmse_orig,
        'r2_orig': r2_orig
    })

# Summary Statistics
print(f"\n{'='*70}")
print(f"SUMMARY: {K_FOLDS}-FOLD CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

results_df = pd.DataFrame(fold_results)

print("\n" + "="*70)
print("PERFORMANCE FOR EACH FOLD:")
print("="*70)

for idx, row in results_df.iterrows():
    print(f"\nFold {row['fold']}:")
    print(f"  Training samples: {row['n_train']:,}")
    print(f"  Test samples:     {row['n_test']:,}")
    print(f"  Log Space Metrics:")
    print(f"    MAE:  {row['mae_log']:.4f}")
    print(f"    MSE:  {row['mse_log']:.4f}")
    print(f"    R²:   {row['r2_log']:.4f}")
    print(f"  Real Space Metrics (µg/L):")
    print(f"    MAE:  {row['mae_orig']:.4f}")
    print(f"    RMSE: {row['rmse_orig']:.4f}")
    print(f"    R²:   {row['r2_orig']:.4f}")

print("\n" + "="*70)
print("AVERAGE PERFORMANCE ACROSS ALL FOLDS:")
print("="*70)
print(f"\nLog Space Metrics:")
print(f"  MAE:  {results_df['mae_log'].mean():.4f} ± {results_df['mae_log'].std():.4f}")
print(f"  MSE:  {results_df['mse_log'].mean():.4f} ± {results_df['mse_log'].std():.4f}")
print(f"  R²:   {results_df['r2_log'].mean():.4f} ± {results_df['r2_log'].std():.4f}")

print(f"\nReal Space Metrics (µg/L):")
print(f"  MAE:  {results_df['mae_orig'].mean():.4f} ± {results_df['mae_orig'].std():.4f}")
print(f"  RMSE: {results_df['rmse_orig'].mean():.4f} ± {results_df['rmse_orig'].std():.4f}")
print(f"  R²:   {results_df['r2_orig'].mean():.4f} ± {results_df['r2_orig'].std():.4f}")

# Save results
results_df.to_csv('kfold_results.csv', index=False)
print(f"\n{'='*70}")
print("Results saved to 'kfold_results.csv'")
print(f"{'='*70}")