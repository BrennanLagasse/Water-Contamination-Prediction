import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("(Cleaned) WQP Full Physical Chemical.csv")
df2 = pd.read_csv("(Cleaned) WQP + MRDS + gNATSGO.csv")
truth = pd.read_csv("Ground Truth.csv")

df["ActivityStartDate"] = pd.to_datetime(df["ActivityStartDate"])
mask = (
    df.sort_values("ActivityStartDate")
        .groupby(["ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure", "CharacteristicName"])
        .tail(1)
        .index
)

df = df.loc[mask]
df2 = df2.loc[mask]
truth = truth.loc[mask]

print(df["CharacteristicName"].value_counts())
plt.hist(df["ActivityStartDate"].dt.year, bins=40, edgecolor="black")
plt.xlabel("Year")
plt.ylabel("Number of Samples")
plt.title("Dsitribution of Samples across Time")
plt.show()

df2.to_csv("final_dataset.csv", index=False)
truth.to_csv("final_dataset_truth.csv", index=False)