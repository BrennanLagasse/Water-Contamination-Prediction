import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import seaborn as sns
import matplotlib.pyplot as plt
import shap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm

# Load Data
df = pd.read_csv("test2.csv")
truth = pd.read_csv("testTruth.csv")
print(df.columns.tolist())
mask = df["CharacteristicName"] == "Arsenic"
X = df[mask]
y = truth["ResultMeasureValue"][mask]
X = X.drop(["CharacteristicName", "id"], axis=1)

feature_names = X.columns
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

mask = (y >= 0) & (~np.isnan(y)) & (~np.isinf(y))
X = X[mask]
y = y[mask]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print("Median y:", y.median())

X = X.values.astype(np.float32)
y = y.values.astype(np.float32).reshape(-1, 1)
y = np.log1p(y)

# Train/val/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosted Tree
model = XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=8, objective='reg:absoluteerror', eval_metric='mae', n_jobs=-1)
#model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=127, n_jobs=-1)
#model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=16, loss_function="MAE", eval_metric="MAE", random_seed=42, verbose=0, task_type="CPU")

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)]
)
# Validation Set Evaluation
y_val_pred = model.predict(X_val)

# Predicted vs Actual Scatter
plt.figure(figsize=(6,6))
plt.scatter(y_val, y_val_pred, alpha=0.5, label="Predictions")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         'r--', label="x = y (Perfect Prediction)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual in log(x+1) scale (XGBoost, Validation Set)")
plt.legend()
plt.grid(True)
plt.show()

y_val_pred_lin = np.expm1(y_val_pred.flatten())
y_val_true_lin = np.expm1(y_val.flatten())

# Compute residuals in log(x+1) space
error = np.log1p(y_val_pred_lin) - np.log1p(y_val_true_lin)

mask = df["CharacteristicName"] == "Arsenic"
X_full = df[mask].copy()
X_full = X_full.drop(["CharacteristicName", "id"], axis=1)

mask = (truth["ResultMeasureValue"] >= 0) & (~np.isnan(truth["ResultMeasureValue"])) & (~np.isinf(truth["ResultMeasureValue"]))
X_full = X_full[mask]

X_full = X_full.reset_index(drop=True)
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

lat = X_val_df["ActivityLocation/LatitudeMeasure"].values
lon = X_val_df["ActivityLocation/LongitudeMeasure"].values

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-125, -66, 25, 49], crs=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, edgecolor='gray')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4)

# Normalize color scale around 0
vmin = np.min(error)
vmax = np.max(error)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# Scatter plot
sc = ax.scatter(
    lon, lat,
    c=error,
    cmap="RdBu_r",
    norm=norm,
    s=20,
    edgecolor='k',
    linewidth=0.2,
    alpha=0.8,
    transform=ccrs.PlateCarree()
)

# Colorbar
cbar = plt.colorbar(sc, orientation='vertical', pad=0.02, shrink=0.8)
cbar.set_label("Log(x+1) Residual (Predicted − Actual)", fontsize=11)
plt.title("Spatial Distribution of Arsenic Prediction Error in log(x+1) scale(XGBoost, Validation Set)", fontsize=12)
plt.tight_layout()
plt.show()

# SHAP for Feature Importance
print("Running SHAP")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(
    shap_values,
    X_val,
    plot_type="bar",
    max_display=20,
    show=True,
    feature_names=feature_names
)