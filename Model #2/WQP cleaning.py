import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
wqp = pd.read_csv("WQP Full Physical Chemical.csv")
wqp = wqp.dropna(axis=1, how="all")

# Picking important columns
important = wqp[["Location_LatitudeStandardized", "Location_LongitudeStandardized", "Result_Characteristic", "Location_HUCEightDigitCode", "Activity_StartDate", "Result_Measure", "Result_MeasureUnit"]].copy()

# Showing an extremely strong correlation between not detected values and NaN result measures
nan_mask = wqp['Result_Measure'].isna()
maybe_zero_mask = wqp['DetectionLimit_MeasureA'].notna()
maybe_zero_pct = (nan_mask & maybe_zero_mask).sum() / nan_mask.sum() * 100
print(f"Percent of NaNs that might actually be 0s (non-detects): {maybe_zero_pct:.2f}%")

# Adding IsDetected Columns and imputing values in result measure
mask_A = important['Result_Measure'].isna() & wqp['DetectionLimit_MeasureA'].notna()
important.loc[mask_A, 'Result_Measure'] = wqp.loc[mask_A, 'DetectionLimit_MeasureA'] / 2
important.loc[mask_A, "Result_MeasureUnit"] = wqp.loc[mask_A, "DetectionLimit_MeasureUnitA"]

mask_B = important['Result_Measure'].isna() & wqp['DetectionLimit_MeasureA'].isna() & wqp['DetectionLimit_MeasureB'].notna()
important.loc[mask_B, 'Result_Measure'] = wqp.loc[mask_B, 'DetectionLimit_MeasureB'] / 2
important.loc[mask_B, "Result_MeasureUnit"] = wqp.loc[mask_B, "DetectionLimit_MeasureUnitB"]
important['Is_Detected'] = important['Result_Measure'].notna().astype(int)

# Converting all values to ug/L
mask_mg = (important["Result_MeasureUnit"] == "mg/L") | (important["Result_MeasureUnit"] == "mg/kg")
important.loc[mask_mg, "Result_Measure"] *= 1000
important = important.drop("Result_MeasureUnit", axis=1)

# Dropping all NA Values
important = important.dropna(subset=["Location_LatitudeStandardized", "Location_LongitudeStandardized", "Result_Measure", "Location_HUCEightDigitCode"])

# Extracting Ground Truth Labels
ground_truth = important["Result_Measure"]
important = important.drop("Result_Measure", axis=1)
# Final Touches
print(important.isna().mean() * 100)
print(important.head())
important.to_csv("(Cleaned) WQP Full Physical Chemical.csv", index=False)
ground_truth.to_csv("Ground Truth.csv", index=False)