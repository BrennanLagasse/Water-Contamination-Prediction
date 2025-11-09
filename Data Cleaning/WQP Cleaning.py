import pandas as pd
import math
import torch
import torch.nn as nn
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

wqp = pd.read_csv("WQP Physical Chemical.csv", low_memory=False)
station = pd.read_csv("WQP Station Metadata.csv", low_memory=False)

# Dropping useless columns
wqp = wqp.dropna(axis=1, thresh=int(0.1*len(wqp)))
station = station.dropna(axis=1, thresh=int(0.1*len(station)))

# Dropping rows with wrong water locations and missing lat / long
wqp = wqp[wqp["ActivityMediaSubdivisionName"].isin(["Groundwater", "Surface Water", "Ground Water"])]
wqp = wqp[~wqp[['MonitoringLocationIdentifier', 'ActivityLocation/LatitudeMeasure', 'ActivityLocation/LongitudeMeasure']].isnull().all(axis=1)]

# Imputing Lat / Long and selecting relevant columns
wqp = wqp[["MonitoringLocationIdentifier", "CharacteristicName", "ActivityStartDate", "ResultSampleFractionText", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure", "ResultMeasureValue", "ResultMeasure/MeasureUnitCode", "DetectionQuantitationLimitMeasure/MeasureValue", "DetectionQuantitationLimitMeasure/MeasureUnitCode"]].merge(
    station[["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "HUCEightDigitCode", "StateCode", "CountyCode", "WellDepthMeasure/MeasureValue", "WellDepthMeasure/MeasureUnitCode"]],
    on="MonitoringLocationIdentifier",
    how="left",
    suffixes=("", "_lookup")
)
wqp['ActivityLocation/LatitudeMeasure'] = wqp['ActivityLocation/LatitudeMeasure'].fillna(wqp['LatitudeMeasure'])
wqp['ActivityLocation/LongitudeMeasure'] = wqp['ActivityLocation/LongitudeMeasure'].fillna(wqp['LongitudeMeasure'])
wqp = wqp.drop(["LatitudeMeasure", "LongitudeMeasure", "MonitoringLocationIdentifier"], axis=1)

# Changing values to float in result measure and imputing half of limit for values that look like <0.03
def try_convert(x):
    try:
        return float(x)
    except:
        return x
wqp["ResultMeasureValue"] = wqp["ResultMeasureValue"].apply(try_convert)

# Imputing half limit for values in ResultMeasure like <0.03
def impute_below_detection(val):
    if isinstance(val, str) and val.startswith("<"):
        try:
            num = float(val[1:].strip())
            return num / 2
        except ValueError:
            return val
    return val
wqp["ResultMeasureValue"] = wqp["ResultMeasureValue"].apply(impute_below_detection)

# Dropping Bad Values
original_nans = wqp["ResultMeasureValue"].isna()
converted = pd.to_numeric(wqp["ResultMeasureValue"], errors="coerce")
wqp = wqp[original_nans | converted.notna()]
wqp["ResultMeasureValue"] = converted

# Checking type counts
def precise_type(x):
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    return type(x).__name__

type_counts = wqp["ResultMeasureValue"].apply(precise_type).value_counts()
print(type_counts)

# Shows a decent correlation between missing result measure value and detection limit
nan_mask = wqp['ResultMeasureValue'].isna()
print(wqp["ResultMeasureValue"].isna().mean() * 100)
wqp["ResultMeasureValue"].to_csv("testing.csv")
notnan_mask = wqp["ResultMeasureValue"].notna()
maybe_zero_mask = wqp['DetectionQuantitationLimitMeasure/MeasureValue'].notna()
maybe_zero_pct = (nan_mask & maybe_zero_mask).sum() / nan_mask.sum() * 100
other = (notnan_mask & maybe_zero_mask).sum() / notnan_mask.sum() * 100
print(f"Percent of NaNs that might actually be 0s (non-detects): {maybe_zero_pct:.2f}%")
print(f"Other: {other}")

# Imputing half of detection limit
wqp = wqp.dropna(subset=['ResultMeasureValue', 'DetectionQuantitationLimitMeasure/MeasureValue'], how='all')
wqp['DetectionQuantitationLimitMeasure/MeasureValue'] = pd.to_numeric(wqp['DetectionQuantitationLimitMeasure/MeasureValue'], errors='coerce')
wqp['ResultMeasureValue'] = wqp['ResultMeasureValue'].fillna(wqp['DetectionQuantitationLimitMeasure/MeasureValue'] / 2)

# Imputing the corresponding unit for the amount number
filled_mask = wqp['ResultMeasureValue'] == (wqp['DetectionQuantitationLimitMeasure/MeasureValue'] / 2)
wqp.loc[filled_mask, 'ResultMeasure/MeasureUnitCode'] = wqp.loc[filled_mask, 'DetectionQuantitationLimitMeasure/MeasureUnitCode']
wqp = wqp.drop(["DetectionQuantitationLimitMeasure/MeasureValue", "DetectionQuantitationLimitMeasure/MeasureUnitCode"], axis=1)

print(wqp.shape)
print(wqp["CharacteristicName"].value_counts(normalize=True))

# Converting all values to ug/L
wqp = wqp[wqp["ResultMeasure/MeasureUnitCode"].isin(["ug/l", "mg/L", "ug/L", "mg/l", "ug/L as As"])]
mg_mask = wqp["ResultMeasure/MeasureUnitCode"].isin(["mg/L", "mg/l"])
wqp.loc[mg_mask, "ResultMeasureValue"] *= 1000
wqp = wqp.drop("ResultMeasure/MeasureUnitCode", axis=1)

# Filling NA Values
wqp = wqp.dropna(subset=["HUCEightDigitCode"])
print("Missing: " + str(wqp["WellDepthMeasure/MeasureValue"].isna().mean() * 100))
wqp['WellDepthMeasure/MeasureValue'] = wqp.groupby('CountyCode')['WellDepthMeasure/MeasureValue'].transform(
    lambda x: x.fillna(x.median())
)
wqp['WellDepthMeasure/MeasureValue'] = wqp['WellDepthMeasure/MeasureValue'].fillna(
    wqp['WellDepthMeasure/MeasureValue'].median()
)
wqp = wqp.drop("WellDepthMeasure/MeasureUnitCode", axis=1) # All feet

# Embeddings for ResultSampleFractionText, StateCode, CountyCode, and HUCEightDigitCode
embed_columns = {
    "ResultSampleFractionText": 2,
    "StateCode": 5,
    "CountyCode": 8,
    "HUCEightDigitCode": 10
}

for col, emb_dim in embed_columns.items():
    wqp[col] = wqp[col].fillna("Missing").astype(str)

    vocab = {v: i for i, v in enumerate(wqp[col].unique())}
    wqp[f"{col}_idx"] = wqp[col].map(vocab)
    
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=emb_dim)

    idx_tensor = torch.tensor(wqp[f"{col}_idx"].fillna(0).astype(int).values)

    with torch.no_grad():
        vectors = embedding(idx_tensor)

    emb_df = pd.DataFrame(vectors.numpy(), columns=[f"{col}_emb_{i}" for i in range(emb_dim)])
    wqp = pd.concat([wqp.reset_index(drop=True), emb_df], axis=1)
    wqp = wqp.drop([f"{col}_idx", col], axis=1)

# Extracting Ground Truth Labels
ground_truth = wqp["ResultMeasureValue"]
wqp = wqp.drop("ResultMeasureValue", axis=1)

# Rounding rows in ResultMeasure so we don't have 702.000000001 then removing outliers
ground_truth = ground_truth.round(4)

# Final Touches
#print(wqp["ResultSampleFractionText"].value_counts(normalize=True))
print(wqp["CharacteristicName"].value_counts(normalize=True))
print(wqp.isna().mean() * 100)
print(ground_truth.isna().mean() * 100)
print(wqp.head())
print(wqp.shape)
wqp.to_csv("(Cleaned) WQP Full Physical Chemical.csv", index=False)
ground_truth.to_frame().to_csv("Ground Truth.csv", index=False)