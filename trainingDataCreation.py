import pandas as pd
from sklearn.preprocessing import LabelEncoder
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Load the new merged dataset
df = pd.read_csv("mrds_naics_merge.csv")
print(df.head())

# Select feature columns (exclude station_id)
feature_cols = [
    "As-tot", "distance_km_mrds",
    "percentage_Manufacturing", "percentage_Administrative", "percentage_Wholesale Trade"
]
X = df[feature_cols]

# Create binary label based on a threshold for As-tot
y = (df["As-tot"] == 1).astype(int)

# Save to CSV
X.to_csv("Training Data.csv", index=False)
y.to_csv("Training Truth.csv", index=False)