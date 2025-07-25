import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

mrds = pd.read_csv("MRDS Merged.csv")
mrds = mrds.dropna(subset=["wgs84_lat", "wgs84_lon"])
mrds = mrds.dropna(axis=1, thresh=int(0.4*len(mrds)))

mrds = mrds.drop(["commod", "code"], axis=1)
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
mrds.to_csv("(Cleaned) MRDS.csv", index=False)
print(mrds.isna().mean() * 100)