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
print(f"Using device: {device}")

# Load Data
print("\nLoading data...")
df = pd.read_csv("final_dataset_with_huc.csv")
truth = pd.read_csv("final_dataset_truth.csv")
fold_assignments = pd.read_csv("huc4_fold_assignments.csv")

# Filter to Arsenic only
arsenic_mask = df["CharacteristicName"] == "Arsenic"
X = df[arsenic_mask].copy()
y = truth["ResultMeasureValue"][arsenic_mask].copy()
X = X.drop(["CharacteristicName", "id"], axis=1, errors='ignore')

print(f"Initial X shape: {X.shape}")
print(f"Initial y shape: {y.shape}")

# Apply valid value filter
valid_mask = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X = X[valid_mask].reset_index(drop=True)
y = y[valid_mask].reset_index(drop=True)

print(f"After filtering - X shape: {X.shape}, y shape: {y.shape}")
print(f"Median y: {y.median():.4f}")

# Check for missing HUC4 codes
missing_huc = X['HUC4'].isna().sum()
print(f"Samples with missing HUC codes: {missing_huc}")

if missing_huc > 0:
    print(f"Dropping {missing_huc} samples without HUC codes...")
    X = X.dropna(subset=['HUC4']).reset_index(drop=True)
    y = y[X.index].reset_index(drop=True)

# Merge with fold assignments
fold_assignments['HUC4'] = fold_assignments['HUC4'].astype(str).str.zfill(4)
X['HUC4'] = X['HUC4'].astype(str).str.split('.').str[0].str.zfill(4)
X = X.merge(fold_assignments, on='HUC4', how='left')

# Check for missing fold assignments
missing_folds = X['fold'].isna().sum()
if missing_folds > 0:
    print(f"WARNING: {missing_folds} samples have no fold assignment. These will be dropped.")
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

# Building the Graph
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

# Prepare PyG Data
x = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y_array, dtype=torch.float)

# K-Fold Cross-Validation
K_FOLDS = len(np.unique(fold_labels))

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
    
    # Create PyG Data object
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
    
    for epoch in range(500):
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
    
    # Convert to original space
    y_pred_orig = np.expm1(y_pred_test.flatten())
    y_true_orig = np.expm1(y_true_test.flatten())
    
    # Filter for real-space metrics: 0 <= y <= 100 and no negative predictions
    real_space_filter = (y_true_orig >= 0) & (y_true_orig <= 100) & (y_pred_orig >= 0)
    y_pred_filtered = y_pred_orig[real_space_filter]
    y_true_filtered = y_true_orig[real_space_filter]
    
    n_test_filtered = len(y_true_filtered)
    n_test_removed = len(y_true_orig) - n_test_filtered
    
    # Metrics in original space (filtered: 0-100 µg/L, no negative predictions)
    if n_test_filtered > 0:
        mae_orig = mean_absolute_error(y_true_filtered, y_pred_filtered)
        mse_orig = mean_squared_error(y_true_filtered, y_pred_filtered)
        r2_orig = r2_score(y_true_filtered, y_pred_filtered)
        rmse_orig = np.sqrt(mse_orig)
    else:
        mae_orig = mse_orig = r2_orig = rmse_orig = np.nan
        print(f"WARNING: No valid samples for real-space metrics in Fold {fold_idx + 1}")
    
    print(f"\n--- Fold {fold_idx + 1} Results ---")
    print(f"Log Space (all data):  MAE={mae_log:.4f}, MSE={mse_log:.4f}, R²={r2_log:.4f}")
    print(f"Real Space (0-100 µg/L, {n_test_filtered}/{n_test} samples): MAE={mae_orig:.4f} µg/L, RMSE={rmse_orig:.4f} µg/L, R²={r2_orig:.4f}")
    if n_test_removed > 0:
        print(f"  Note: {n_test_removed} samples excluded from real-space metrics (y>100 or negative predictions)")
    
    fold_results.append({
        'fold': fold_idx + 1,
        'n_train': n_train,
        'n_test': n_test,
        'n_test_filtered': n_test_filtered,
        'n_test_removed': n_test_removed,
        'mae_log': mae_log,
        'mse_log': mse_log,
        'r2_log': r2_log,
        'mae_orig': mae_orig,
        'mse_orig': mse_orig,
        'rmse_orig': rmse_orig,
        'r2_orig': r2_orig
    })

results_df = pd.DataFrame(fold_results)

print("PERFORMANCE FOR EACH FOLD:")

for idx, row in results_df.iterrows():
    print(f"\nFold {row['fold']}:")
    print(f"  Training samples: {row['n_train']:,}")
    print(f"  Test samples:     {row['n_test']:,} (filtered: {row['n_test_filtered']:,}, removed: {row['n_test_removed']:,})")
    print(f"  Log Space Metrics (all data):")
    print(f"    MAE:  {row['mae_log']:.4f}")
    print(f"    MSE:  {row['mse_log']:.4f}")
    print(f"    R²:   {row['r2_log']:.4f}")
    print(f"  Real Space Metrics (0-100 µg/L only):")
    print(f"    MAE:  {row['mae_orig']:.4f} µg/L")
    print(f"    RMSE: {row['rmse_orig']:.4f} µg/L")
    print(f"    R²:   {row['r2_orig']:.4f}")

print("\n" + "="*70)
print("AVERAGE PERFORMANCE ACROSS ALL FOLDS:")
print("="*70)
print(f"\nLog Space Metrics (all data):")
print(f"  MAE:  {results_df['mae_log'].mean():.4f} ± {results_df['mae_log'].std():.4f}")
print(f"  MSE:  {results_df['mse_log'].mean():.4f} ± {results_df['mse_log'].std():.4f}")
print(f"  R²:   {results_df['r2_log'].mean():.4f} ± {results_df['r2_log'].std():.4f}")

print(f"\nReal Space Metrics (0-100 µg/L only):")
print(f"  MAE:  {results_df['mae_orig'].mean():.4f} ± {results_df['mae_orig'].std():.4f} µg/L")
print(f"  RMSE: {results_df['rmse_orig'].mean():.4f} ± {results_df['rmse_orig'].std():.4f} µg/L")
print(f"  R²:   {results_df['r2_orig'].mean():.4f} ± {results_df['r2_orig'].std():.4f}")

print(f"\nTotal samples removed across all folds: {results_df['n_test_removed'].sum():,}")

# Save results
results_df.to_csv('gat_kfold_results.csv', index=False)
print("Results saved to 'gat_kfold_results.csv'")