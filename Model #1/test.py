import pandas as pd
df = pd.read_csv("MRDS US Mineral Chemicals.csv")
df["CODE_LIST"] = df["CODE_LIST"].apply(
    lambda x: " ".join([chem.capitalize() + "-tot" for chem in str(x).split()])
)
df.to_csv("MRDS US Mineral Chemicals v2.csv", index=False)