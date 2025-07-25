import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import BallTree

# Load data
wqp_df = pd.read_csv("(Cleaned) WQP Full Physical Chemical.csv").drop("AquiferName", axis=1)
mrds_df = pd.read_csv("(Cleaned) MRDS.csv")

# Coordinates
wqp_coords = np.radians(wqp_df[["ActivityLocation/LatitudeMeasure", "ActivityLocation/LongitudeMeasure"]].values)
mrds_coords = np.radians(mrds_df[["wgs84_lat", "wgs84_lon"]].values)

# BallTree
tree = BallTree(mrds_coords, metric='haversine')
radius_km = 3
radius_rad = radius_km / 6371.0
indices = tree.query_radius(wqp_coords, r=radius_rad)

# Textual columns (non-coordinate)
exclude_cols = {"dep_id", "wgs84_lat", "wgs84_lon"}
embed_cols = [col for col in mrds_df.columns if col not in exclude_cols]

# Build vocab/token lists
vocabularies = {}
tokenized = {}
for col in embed_cols:
    vocab = {}
    tokens_col = []
    for val in mrds_df[col].fillna(""):
        tokens = [t.strip().lower() for t in str(val).split(",") if t.strip()]
        tokens_col.append(tokens)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    vocabularies[col] = vocab
    tokenized[col] = tokens_col

# Create embedding layers (fixed dim for simplicity)
embedding_dims = {col: 4 for col in embed_cols}
embedding_layers = {col: nn.Embedding(len(vocabularies[col]), embedding_dims[col]) for col in embed_cols}

# Compute embedding vectors for each WQP point
agg_embeddings = []
for i, matched_indices in enumerate(indices):
    if i % 1000 == 0:
        print(f"Processing row {i} / {len(indices)}")
    row_vectors = []
    for col in embed_cols:
        emb_layer = embedding_layers[col]
        vocab = vocabularies[col]
        tokens = []
        for idx in matched_indices:
            tokens.extend(tokenized[col][idx])
        if tokens:
            ids = torch.tensor([vocab[t] for t in tokens if t in vocab])
            emb = emb_layer(ids)
            row_vectors.append(emb.mean(dim=0))
        else:
            row_vectors.append(torch.zeros(embedding_dims[col]))
    agg_embeddings.append(torch.cat(row_vectors))

# Convert to dataframe
embedding_tensor = torch.stack(agg_embeddings)  # shape: (num_wqp, total_dim)
embedding_np = embedding_tensor.detach().numpy()

# Generate column names like gold_0, gold_1, ..., copper_0, ...
column_names = []
for col in embed_cols:
    for i in range(embedding_dims[col]):
        column_names.append(f"{col}_emb_{i}")

embedding_df = pd.DataFrame(embedding_np, columns=column_names)

# Combine and save
combined_df = pd.concat([wqp_df.reset_index(drop=True), embedding_df], axis=1)
combined_df.to_csv("WQP + MRDS.csv", index=False)
print("Saved to: WQP + MRDS (Embedded).csv")
