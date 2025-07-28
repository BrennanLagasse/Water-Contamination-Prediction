import pandas as pd
import torch.nn as nn
import torch
from sklearn.preprocessing import MultiLabelBinarizer

mrds = pd.read_csv("MRDS Merged.csv")
mrds = mrds.dropna(subset=["wgs84_lat", "wgs84_lon"])
mrds = mrds.dropna(axis=1, thresh=int(0.7*len(mrds)))

"""
# One Hot Encoding
split_series = mrds['commod_group'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

split_series = mrds['phys_div'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

split_series = mrds['commod_tp'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

split_series = mrds['import'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

split_series = mrds['land_st'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

mrds[["phys_prov", "phys_sect"]] = mrds[["phys_prov", "phys_sect"]].fillna("Missing")

split_series = mrds['phys_prov'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

split_series = mrds['phys_sect'].fillna('').astype(str).str.split(', ').apply(lambda x: [i.strip() for i in x])
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(split_series), columns=mlb.classes_, index=mrds.index)
mrds = pd.concat([mrds, one_hot], axis=1)

mrds = mrds.drop(["commod_tp", "phys_div", "commod_group", "import", "land_st", "phys_prov", "phys_sect"], axis=1)"""

# Embeddings for first 3 columns
cols = ["commod_tp", "commod_group_x", "import_x"]
for col in cols:
    vocab = {}
    tokenized = []

    for val in mrds[col].fillna(""):
        tokens = [t.strip().lower() for t in str(val).split(",") if t.strip()]
        tokenized.append(tokens)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    index_lists = []
    for tokens in tokenized:
        index_lists.append(sorted([vocab[t] for t in tokens]))
    mrds[col + "_idx"] = index_lists

embedding_dims = {
    "commod_tp": 4,
    "commod_group_x": 4,
    "import_x": 2
}

embedding_layers = {}
for col in cols:
    vocab_size = max(max(lst) if lst else -1 for lst in mrds[col + "_idx"]) + 1
    emb_dim = embedding_dims[col]
    embedding_layers[col] = nn.Embedding(vocab_size, emb_dim)

for col in cols:
    emb_layer = embedding_layers[col]
    emb_dim = embedding_dims[col]

    embedded_vectors = []
    for idx_list in mrds[col + "_idx"]:
        if len(idx_list) == 0:
            embedded = torch.zeros(emb_dim)
        else:
            indices = torch.tensor(idx_list)
            with torch.no_grad():
                embedded = emb_layer(indices).mean(dim=0)
        embedded_vectors.append(embedded.numpy())

    emb_df = pd.DataFrame(embedded_vectors, columns=[f"{col}_emb_{i}" for i in range(emb_dim)])
    mrds = pd.concat([mrds.reset_index(drop=True), emb_df], axis=1)

# Embeddings for phys_div (22 unique values), phys_sect (44 unique values)
phys_div_vocab = {val: i for i, val in enumerate(mrds["phys_div"].dropna().unique())}
phys_prov_vocab = {val: i for i, val in enumerate(mrds["phys_prov"].dropna().unique())}

mrds["phys_div_idx"] = mrds["phys_div"].map(phys_div_vocab)
mrds["phys_prov_idx"] = mrds["phys_prov"].map(phys_prov_vocab)

phys_div_emb = nn.Embedding(num_embeddings=22, embedding_dim=4)
phys_prov_emb = nn.Embedding(num_embeddings=44, embedding_dim=6)

phys_div_tensor = torch.tensor(mrds["phys_div_idx"].fillna(0).astype(int).values)
phys_prov_tensor = torch.tensor(mrds["phys_prov_idx"].fillna(0).astype(int).values)

with torch.no_grad():
    phys_div_vecs = phys_div_emb(phys_div_tensor)
    phys_sect_vecs = phys_prov_emb(phys_prov_tensor)

phys_div_df = pd.DataFrame(phys_div_vecs.numpy(), columns=[f"phys_div_emb_{i}" for i in range(phys_div_vecs.shape[1])])
phys_prov_df = pd.DataFrame(phys_sect_vecs.numpy(), columns=[f"phys_prov_emb_{i}" for i in range(phys_sect_vecs.shape[1])])
mrds = pd.concat([mrds.reset_index(drop=True), phys_div_df, phys_prov_df], axis=1)

# Final Touches
mrds = mrds.drop(["commod_tp", "commod_tp_idx", "commod_group_x_idx", "import_x_idx", "commod_group_x", "import_x", "phys_div", "phys_prov", "phys_div_idx", "phys_prov_idx"], axis=1)
print(mrds.isna().mean() * 100)
mrds.to_csv("(Cleaned) MRDS.csv", index=False)