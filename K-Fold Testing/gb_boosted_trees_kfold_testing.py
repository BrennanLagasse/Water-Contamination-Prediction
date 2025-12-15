import pandas as pd
import numpy as np
from xgboost import XGBRegressor
#from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#import seaborn as sns
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load Data
print("\nLoading data")
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

# Store original y values for later
y_orig = y.copy()

# Log-transform target
y_log = np.log1p(y)

# Convert to arrays
X_array = X.values.astype(np.float32)
y_log_array = y_log.values.astype(np.float32)

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
    
    X_train = X_array[train_mask]
    X_test = X_array[test_mask]
    y_train_log = y_log_array[train_mask]
    y_test_log = y_log_array[test_mask]
    
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"Train samples: {n_train:,} ({n_train/len(X_array)*100:.1f}%)")
    print(f"Test samples:  {n_test:,} ({n_test/len(X_array)*100:.1f}%)")
    
    # Initialize and train model
    model = XGBRegressor(
        n_estimators=2000, 
        learning_rate=0.05, 
        max_depth=6, 
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    #model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=127, n_jobs=-1, random_state=RANDOM_SEED)
    #model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=16, loss_function="MAE", eval_metric="MAE", random_seed=RANDOM_SEED, verbose=0, task_type="CPU")
    
    print("Training model...")
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_test, y_test_log)],
        verbose=False
    )
    
    # Predictions on test set (log space)
    y_pred_test_log = model.predict(X_test)
    
    # Metrics in log space (ALL DATA)
    mae_log = mean_absolute_error(y_test_log, y_pred_test_log)
    mse_log = mean_squared_error(y_test_log, y_pred_test_log)
    r2_log = r2_score(y_test_log, y_pred_test_log)
    
    # Convert to original space
    y_pred_orig = np.expm1(y_pred_test_log)
    y_true_orig = y_orig.values[test_mask]
    
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
    
    # Plots for each fold
    # plt.figure(figsize=(6,6))
    # plt.scatter(y_true_filtered, y_pred_filtered, alpha=0.5, label="Predictions")
    # plt.plot([y_true_filtered.min(), y_true_filtered.max()], 
    #          [y_true_filtered.min(), y_true_filtered.max()],
    #          'r--', label="x = y (Perfect Prediction)")
    # plt.xlabel("Actual Values (µg/L)")
    # plt.ylabel("Predicted Values (µg/L)")
    # plt.title(f"Fold {fold_idx + 1}: Predicted vs Actual (Test Set)")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'fold_{fold_idx+1}_predictions.png')
    # plt.close()

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
results_df.to_csv('xgboost_kfold_results.csv', index=False)
print("Results saved to 'xgboost_kfold_results.csv'")