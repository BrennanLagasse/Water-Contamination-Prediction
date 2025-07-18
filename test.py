import pandas as pd

mrds = pd.read_csv("MRDS v2 Cleaned.csv")
station = pd.read_csv("GEMS Station Metadata.csv")

print(mrds.dtypes.value_counts(normalize=True) * 100)
print(station.dtypes.value_counts(normalize=True) * 100)