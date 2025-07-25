import pandas as pd
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
wqp = pd.read_csv("WQP Physical Chemical.csv")
station = pd.read_csv("WQP Station Metadata.csv")
wqp = wqp.dropna(axis=1, thresh=int(0.1*len(wqp)))
station = station.dropna(axis=1, thresh=int(0.1*len(station)))
wqp = wqp[wqp["ActivityMediaSubdivisionName"].isin(["Groundwater", "Surface Water", "Ground Water"])]
wqp = wqp[~wqp[['MonitoringLocationIdentifier', 
             'ActivityLocation/LatitudeMeasure', 
             'ActivityLocation/LongitudeMeasure']].isnull().all(axis=1)]
wqp = wqp[["MonitoringLocationIdentifier", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure", "ResultMeasureValue", "ResultMeasure/MeasureUnitCode", "DetectionQuantitationLimitMeasure/MeasureValue", "DetectionQuantitationLimitMeasure/MeasureUnitCode"]].merge(
    station[["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "HUCEightDigitCode", "StateCode", "CountyCode", "WellDepthMeasure/MeasureValue", "WellDepthMeasure/MeasureUnitCode", "AquiferName"]],
    on="MonitoringLocationIdentifier",
    how="left",
    suffixes=("", "_lookup")
)
wqp['ActivityLocation/LatitudeMeasure'] = wqp['ActivityLocation/LatitudeMeasure'].fillna(wqp['LatitudeMeasure'])
wqp['ActivityLocation/LongitudeMeasure'] = wqp['ActivityLocation/LongitudeMeasure'].fillna(wqp['LongitudeMeasure'])
wqp = wqp.drop(["LatitudeMeasure", "LongitudeMeasure", "MonitoringLocationIdentifier"], axis=1)

# Shows a decent correlation between missing result measure value and detection limit
nan_mask = wqp['ResultMeasureValue'].isna()
notnan_mask = wqp["ResultMeasureValue"].notna()
maybe_zero_mask = wqp['DetectionQuantitationLimitMeasure/MeasureValue'].notna()
maybe_zero_pct = (nan_mask & maybe_zero_mask).sum() / nan_mask.sum() * 100
other = (notnan_mask & maybe_zero_mask).sum() / notnan_mask.sum() * 100
print(f"Percent of NaNs that might actually be 0s (non-detects): {maybe_zero_pct:.2f}%")
print(f"Other: {other}")

# Imputing half of detection limit
wqp = wqp.dropna(subset=[
    'ResultMeasureValue', 
    'DetectionQuantitationLimitMeasure/MeasureValue'
], how='all')
wqp['DetectionQuantitationLimitMeasure/MeasureValue'] = pd.to_numeric(
    wqp['DetectionQuantitationLimitMeasure/MeasureValue'], errors='coerce'
)
wqp['ResultMeasureValue'] = wqp['ResultMeasureValue'].fillna(
    wqp['DetectionQuantitationLimitMeasure/MeasureValue'] / 2
)
filled_mask = wqp['ResultMeasureValue'] == (wqp['DetectionQuantitationLimitMeasure/MeasureValue'] / 2)
wqp.loc[filled_mask, 'ResultMeasure/MeasureUnitCode'] = wqp.loc[filled_mask, 'DetectionQuantitationLimitMeasure/MeasureUnitCode']
wqp = wqp.drop(["DetectionQuantitationLimitMeasure/MeasureValue", "DetectionQuantitationLimitMeasure/MeasureUnitCode"], axis=1)
print(wqp.shape)

# Converting all values to ug/L
wqp = wqp[wqp["ResultMeasure/MeasureUnitCode"].isin(["ug/l", "mg/L", "ug/L", "mg/l", "ug/L as As"])]
mg_mask = wqp["ResultMeasure/MeasureUnitCode"].isin(["mg/L", "mg/l"])
wqp.loc[mg_mask, "ResultMeasureValue"] *= 1000
wqp = wqp.drop("ResultMeasure/MeasureUnitCode", axis=1)

# Filling NA Values
wqp = wqp.dropna(subset=["HUCEightDigitCode"])
wqp['WellDepthMeasure/MeasureValue'] = wqp.groupby('CountyCode')['WellDepthMeasure/MeasureValue'].transform(
    lambda x: x.fillna(x.median())
)
wqp['WellDepthMeasure/MeasureValue'] = wqp['WellDepthMeasure/MeasureValue'].fillna(
    wqp['WellDepthMeasure/MeasureValue'].median()
)
wqp = wqp.drop("WellDepthMeasure/MeasureUnitCode", axis=1) # All feet
wqp["AquiferName"] = wqp["AquiferName"].fillna("Missing")
wqp = pd.get_dummies(wqp, columns=["AquiferName"])
print(wqp.isna().mean() * 100)

# Extracting Ground Truth Labels
ground_truth = wqp["ResultMeasureValue"]
wqp = wqp.drop("ResultMeasureValue", axis=1)

# Dropping Bugged Rows ("0.00590.0.0059..")
converted = pd.to_numeric(ground_truth, errors="coerce")
mask = converted.notna()
num_invalid = (~mask).sum()
print(f"Number of rows that can't be converted to float: {num_invalid}")
ground_truth = converted[mask].copy()
wqp = wqp[mask].copy()

# Final Touches
print(wqp.isna().mean() * 100)
print(wqp.head())
print(wqp.shape)
wqp.to_csv("(Cleaned) WQP Full Physical Chemical.csv", index=False)
ground_truth.to_frame().to_csv("Ground Truth.csv", index=False)