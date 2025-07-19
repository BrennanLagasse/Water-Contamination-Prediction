import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Display full data
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# === Load All Files ===
original = pd.read_csv("(Cleaned) WQP Full Physical Chemical.csv")
truth = pd.read_csv("Ground Truth.csv")
merged_mukey = pd.read_csv("WQP + SSURGO (Mukey Only).csv")
component = pd.read_csv("component.csv")
chorizon = pd.read_csv("chorizon.csv")

print(f"Original Shape: {original.shape}")
print(f"SSURGO-Added Shape: {merged_mukey.shape}")

# === Drop Useless Columns ===
useless_cols = [
    "Result_Measure", "SampleCollectionMethod_Name",
    "ResultAnalyticalMethod_Name", "OBJECTID", "AREASYMBOL",
    "SPATIALVER", "Shape_Length", "Shape_Area"
]
merged_mukey = merged_mukey.drop(columns=useless_cols, errors="ignore")
merged_mukey = merged_mukey.dropna(subset=["Result_Characteristic"])

# === Standardize Dates ===
original["Activity_StartDate"] = pd.to_datetime(original["Activity_StartDate"])
merged_mukey["Activity_StartDate"] = pd.to_datetime(merged_mukey["Activity_StartDate"])
merged_mukey.to_csv("temp.csv", index=False)
# === Merge to match row order ===
match_cols = [
    "Location_LatitudeStandardized", "Location_LongitudeStandardized",
    "Result_Characteristic", "Location_HUCEightDigitCode",
    "Activity_StartDate", "Is_Detected"
]

merged = original.merge(
    merged_mukey,
    on=match_cols,
    how="left",
    suffixes=("", "_ssurgo")
)

# === Add Cokeys from Components ===
merged = merged.merge(
    component[["mukey", "cokey", "comppct_r"]],
    left_on="MUKEY", right_on="mukey",
    how="left"
)

# === Add Soil Data from Chorizon ===
soil_cols = [
    "ph1to1h2o_r", "cec7_r", "caco3_r", "om_r",
    "sandtotal_r", "claytotal_r", "ksat_r",
    "awc_r", "ec_r", "dbovendry_r", "cokey"
]

merged = merged.merge(chorizon[soil_cols], on="cokey", how="left")

# === Compute Weighted Average of Soil Data by Lat/Long ===
def weighted_mean_ignore_nan(group, col):
    values = group[col]
    weights = group["comppct_r"]
    mask = ~values.isna()
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])

soil_features = [col for col in soil_cols if col != "cokey"]

average_soil = merged.groupby(
    ["Location_LatitudeStandardized", "Location_LongitudeStandardized"]
).apply(
    lambda g: pd.Series({col: weighted_mean_ignore_nan(g, col) for col in soil_features})
).reset_index()

# === Merge back to final output ===
final = original.merge(
    average_soil,
    on=["Location_LatitudeStandardized", "Location_LongitudeStandardized"],
    how="left"
)

# === Dropping Too Empty Rows (Unfortunately 30% dropped) and Converting all columns to numbers ===
mask = final.notna().mean(axis=1) >= 0.8
final = final[mask].reset_index(drop=True)
truth = truth[mask].reset_index(drop=True)

not_filled_columns = ["ph1to1h2o_r", "cec7_r", "caco3_r", "ec_r", "dbovendry_r"]
for col in not_filled_columns:
    final[col] = final[col].fillna(final[col].mean())

final = pd.get_dummies(final, columns=["Result_Characteristic"], dtype=int)
final["Activity_StartDate"] = pd.to_datetime(final["Activity_StartDate"])

# Extract numeric features
final["year"] = final["Activity_StartDate"].dt.year
final["month"] = final["Activity_StartDate"].dt.month
final["day"] = final["Activity_StartDate"].dt.day
final = final.drop("Activity_StartDate", axis=1)

# === Output Final CSV ===
final.to_csv("WQP + SSURGO.csv", index=False)
print(f"Final Shape: {final.shape}")
percent_filled = (final.dropna().shape[0] / final.shape[0]) * 100
print(final.isna().mean() * 100)
filled_80 = (final.notna().mean(axis=1) >= 0.8).mean() * 100
print(f"{filled_80:.2f}% of rows are at least 80% filled")
