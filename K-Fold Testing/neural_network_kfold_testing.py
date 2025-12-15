import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
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

# Standardize features
print("\nStandardizing features...")
X_array = X.values.astype(np.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_array)
y_array = y.values.astype(np.float32).reshape(-1, 1)

print(f"Number of features: {X_scaled.shape[1]}")

# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# K-Fold Cross-Validation
K_FOLDS = len(np.unique(fold_labels))

# Storage for results
fold_results = []

for fold_idx in range(K_FOLDS):
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/{K_FOLDS}")
    print(f"{'='*70}")
    
    # Create train/test masks based on fold assignment
    test_mask = fold_labels == fold_idx
    train_mask = fold_labels != fold_idx
    
    n_train = train_mask.sum()
    n_test = test_mask.sum()
    print(f"Train samples: {n_train:,} ({n_train/len(X_scaled)*100:.1f}%)")
    print(f"Test samples:  {n_test:,} ({n_test/len(X_scaled)*100:.1f}%)")
    
    # Create datasets and dataloaders
    X_train = torch.tensor(X_scaled[train_mask], dtype=torch.float32)
    y_train = torch.tensor(y_array[train_mask], dtype=torch.float32)
    X_test = torch.tensor(X_scaled[test_mask], dtype=torch.float32)
    y_test = torch.tensor(y_array[test_mask], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Initialize model
    model = MLP(input_dim=X_scaled.shape[1]).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    best_train_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in range(500):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
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
    y_pred_test_list = []
    y_true_test_list = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            y_pred_test_list.append(output.cpu().numpy())
            y_true_test_list.append(y_batch.numpy())
    
    y_pred_test = np.vstack(y_pred_test_list).flatten()
    y_true_test = np.vstack(y_true_test_list).flatten()
    
    # Metrics in log space (ALL DATA)
    mae_log = mean_absolute_error(y_true_test, y_pred_test)
    mse_log = mean_squared_error(y_true_test, y_pred_test)
    r2_log = r2_score(y_true_test, y_pred_test)
    
    # Convert to original space
    y_pred_orig = np.expm1(y_pred_test)
    y_true_orig = np.expm1(y_true_test)
    
    # Filter for real-space metrics: 0 <= y <= 100 and no negative predictions
    real_space_filter = (y_true_orig >= 0) & (y_true_orig <= 100) & (y_pred_orig >= 0)
    y_pred_filtered = y_pred_orig[real_space_filter]
    y_true_filtered = y_true_orig[real_space_filter]
    
    n_test_filtered = len(y_true_filtered)
    n_test_removed = len(y_true_orig) - n_test_filtered
    
    # Metrics in original space (FILTERED: 0-100 µg/L, no negative predictions)
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

# Summary Statistics
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
results_df.to_csv('nn_kfold_results.csv', index=False)
print("Results saved to 'nn_kfold_results.csv'")