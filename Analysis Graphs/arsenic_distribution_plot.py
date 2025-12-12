import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading in datasets
df = pd.read_csv("final_dataset.csv", usecols=["CharacteristicName", "ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"])
truth = pd.read_csv("final_dataset_truth.csv")
mask = df["CharacteristicName"] == "Arsenic"
df = df[mask]
truth = truth[mask]

mask = (truth["ResultMeasureValue"] >= 0)
truth = truth[mask]
df = df[mask]
print(f"Truth shape: {truth.shape}")

# Distribution of arsenic values in log scale
plt.hist(np.log1p(truth["ResultMeasureValue"]), bins=30, edgecolor="black")
plt.title("Levels of Arsenic Observed in Samples (log(1+x) µg/L)")
plt.xlabel("Log(1 + As concentration) [µg/L]")
plt.ylabel("Frequency")
plt.show()