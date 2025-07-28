import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree

# Load data
wqp_df = pd.read_csv("(Cleaned) WQP Full Physical Chemical.csv")
mrds_df = pd.read_csv("(Cleaned) MRDS.csv")

# Coordinates
wqp_coords = np.radians(wqp_df[["ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"]].values)
mrds_coords = np.radians(mrds_df[["wgs84_lat", "wgs84_lon"]].values)

# BallTree
tree = BallTree(mrds_coords, metric='haversine')
radius_km = 5
radius_rad = radius_km / 6371.0
indices = tree.query_radius(wqp_coords, r=radius_rad)

embed_cols = [col for col in mrds_df.columns if "_emb_" in col]
aggregated_embeddings = []

for i, idx_list in enumerate(indices):
    if i % 1000 == 0:
        print(f"Row: {i}")
    if len(idx_list) == 0:
        aggregated_embeddings.append(np.zeros(len(embed_cols)))
    else:
        mean_vector = mrds_df.iloc[idx_list][embed_cols].mean().values
        aggregated_embeddings.append(mean_vector)

embedding_df = pd.DataFrame(aggregated_embeddings, columns=[f"mrds_{col}" for col in embed_cols])
combined_df = pd.concat([wqp_df.reset_index(drop=True), embedding_df], axis=1)

combined_df.to_csv("WQP + MRDS.csv", index=False)