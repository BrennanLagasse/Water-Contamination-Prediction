import pandas as pd
import matplotlib.pyplot as plt

# Note: This file is created after running "WQP Cleaning" in the folder "Data Cleaning"
df = pd.read_csv("(Cleaned) WQP Full Physical Chemical.csv")

df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"])
mask = (
    df.sort_values("ActivityStartDate")
        .groupby(["ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure", "CharacteristicName"])
        .tail(1)
        .index
)

df = df.loc[mask]

plt.hist(df["ActivityStartDate"].dt.year, bins=40, edgecolor="black")
plt.xlabel("Year")
plt.ylabel("Number of Samples")
plt.title("Distribution of Samples across Time")
plt.show()