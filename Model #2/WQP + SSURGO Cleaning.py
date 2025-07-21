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

# === Dropping Too Empty Rows (Unfortunately 30% dropped) ===
mask = final.notna().mean(axis=1) >= 0.8
final = final[mask].reset_index(drop=True)
truth = truth[mask].reset_index(drop=True)

# === Converting all columns to numbers ===
not_filled_columns = ["ph1to1h2o_r", "cec7_r", "caco3_r", "ec_r", "dbovendry_r"]
for col in not_filled_columns:
    final[col] = final[col].fillna(final[col].mean())

final = pd.get_dummies(final, columns=["Result_Characteristic"], dtype=int)
final["Activity_StartDate"] = pd.to_datetime(final["Activity_StartDate"])

final["year"] = final["Activity_StartDate"].dt.year
final["month"] = final["Activity_StartDate"].dt.month
final["day"] = final["Activity_StartDate"].dt.day
final = final.drop("Activity_StartDate", axis=1)

# === Removing extreme outliers ===
percent_above = (truth > 1000).mean() * 100
print(f"Percent above 1000: {percent_above}")
mask = truth.values.flatten() <= 1000
truth = truth[mask].reset_index(drop=True)
final = final[mask].reset_index(drop=True)

# === Output Final CSV ===
final.to_csv("WQP + SSURGO.csv", index=False)
truth.to_csv("Ground Truth.csv", index=False)
print(f"Final Shape: {final.shape}")