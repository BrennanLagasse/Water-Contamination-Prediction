import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print("Read 1")
wqp_mrds = pd.read_csv("WQP + MRDS + gNATSGO (MUKEY Only).csv", dtype=str)
chorizon_cols = ["ph1to1h2o_r", "om_r", "claytotal_r", "sandtotal_r", "silttotal_r", "ksat_r", "awc_r", "ec_r", "sar_r", "caco3_r", "kwfact", "kffact", "dbthirdbar_r", "cokey"]
print("Read 2")
chorizon = pd.read_csv("chorizon.csv", usecols=chorizon_cols, dtype=str) # Shortened already for speed so just selects all columns
#chorizon = chorizon.dropna(axis=1, thresh=int(0.8 * len(chorizon)))
component_cols = ["comppct_r", "majcompflag", "slope_r", "hydricrating", "drainagecl", "elev_r", "airtempa_r", "map_r", "ffd_r", "taxsubgrp", "taxorder", "taxsuborder", "taxpartsize", "taxtempregime", "mukey", "cokey"]
print("Read 3")
component = pd.read_csv("component.csv", usecols=component_cols, dtype=str)

# Assigning unique id to each row for grouping later on
wqp_mrds["id"] = range(len(wqp_mrds))
for col in wqp_mrds.columns:
    if col not in ["mukey", "CharacteristicName"]:
        wqp_mrds[col] = wqp_mrds[col].astype(float)
    
# Changing the Cokey format
chorizon["cokey"] = chorizon["cokey"].apply(lambda x: x.split(":")[1] if ":" in x else x)
component["cokey"] = component["cokey"].apply(lambda x: x.split(":")[1] if ":" in x else x)

# Cleaning component - One Hot Encoding and Embeddings for the columns with lots of values
component = pd.get_dummies(component, columns=["majcompflag", "hydricrating", "drainagecl", "taxorder", "taxtempregime"])
for col in ["taxsuborder", "taxsubgrp", "taxpartsize"]:
    vocab = {}
    next_idx = 0
    for word in component[col]:
        if word not in vocab:
            vocab[word] = next_idx
            next_idx += 1
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim = 4)
    def embed_and_split(word):
        idx = torch.tensor(vocab[word])
        vec = embedding(idx)
        return pd.Series(vec.tolist())

    component[[f"{col}_embed_{i}" for i in range(4)]] = component[col].apply(embed_and_split)
component = component.drop(["taxsuborder", "taxsubgrp", "taxpartsize"], axis=1)
for col in component.columns:
    if col not in ["mukey", "cokey"]:
        component[col] = component[col].astype(float)

# Merging to map MUKEYs to COKEYs
merged = wqp_mrds.merge(component, on="mukey", how="left")

# Grouping chorizon by cokey
for col in chorizon.columns:
    if col != "cokey":
        chorizon[col] = chorizon[col].astype(float)

chorizon = chorizon.groupby("cokey", as_index=False).mean(numeric_only=True)

# Merging all
print("Merging chorizon")
print("Unique cokeys in chorizon:", chorizon["cokey"].nunique())
print("Total rows in chorizon:", len(chorizon))
print("Total rows in merged before:", len(merged))

merged = merged.merge(chorizon, on="cokey", how="left")
merged.to_csv("merged1.csv")

# Dealing with NA Values
print("Hi")
for col in ["slope_r", "elev_r", "airtempa_r", "map_r", "ffd_r", "sandtotal_r", "silttotal_r", "claytotal_r", "om_r", "dbthirdbar_r", "ksat_r", "awc_r", "kwfact", "kffact", "caco3_r", "sar_r", "ec_r", "ph1to1h2o_r"]:
    merged[col] = merged[col].astype(float).fillna(merged[col].median())
print("Bye")

# Making everything in merged a float
print(merged.columns)
for col in merged.drop(["mukey", "cokey", "CharacteristicName"], axis=1).columns:
    try:
        merged[col] = merged[col].astype(float)
    except ValueError:
        pass

print(merged.dtypes)
print(merged.shape)
print(merged.head())
# Grouping rows with multiple COKEYs
numeric_part = merged.drop(columns=["mukey", "cokey", "CharacteristicName"]).groupby("id", as_index=False).mean(numeric_only=True)
non_numeric_cols = merged[["id", "mukey", "cokey", "CharacteristicName"]].drop_duplicates(subset="id")
merged = pd.merge(non_numeric_cols, numeric_part, on="id", how="inner")
merged.to_csv("merged2.csv")

print(merged.shape)
print(merged.head())
# KNN to fill in blank MUKEY values
lat_col = "ActivityLocation/LatitudeMeasure"
lon_col = "ActivityLocation/LongitudeMeasure"

missing = merged[merged["mukey"].isna()].copy()
filled = merged[merged["mukey"].notna()].copy()
print("KNN Start")
nn = NearestNeighbors(n_neighbors=1)
nn.fit(filled[[lat_col, lon_col]])

distances, indices = nn.kneighbors(missing[[lat_col, lon_col]])
start_idx = merged.columns.get_loc("mukey")
columns_to_fill = merged.columns[start_idx:].tolist()

for i, idx in enumerate(indices.flatten()):
    nearest_row = filled.iloc[idx]
    for col in columns_to_fill:
        missing.iat[i, missing.columns.get_loc(col)] = nearest_row[col]

# Concatenate back
print("Concatting")
merged = pd.concat([filled, missing], ignore_index=True)
merged.to_csv("merged3.csv")
merged = merged.sort_values(by="id")
merged = merged.drop(["mukey", "cokey"], axis=1)

print(merged.isna().mean() * 100)
print(merged.shape)
merged.to_csv("(Cleaned) WQP + MRDS + gNATSGO.csv", index=False)