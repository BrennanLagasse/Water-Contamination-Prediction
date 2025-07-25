import pandas as pd
pd.set_option("display.max_columns", None)
import os
print("Saving to:", os.getcwd())

ages = pd.read_csv("MRDS/Ages.csv")
ages = ages[["dep_id", "age_tp", "age_young_cd", "age_young"]]
commodity = pd.read_csv("MRDS/Commodity.csv")
commodity = commodity[["dep_id", "commod", "code", "commod_tp", "commod_group", "import"]]
coords = pd.read_csv("MRDS/Coords.csv")
coords = coords[["dep_id", "wgs84_lat", "wgs84_lon"]]
land_status = pd.read_csv("MRDS/Land_status.csv")
land_status = land_status[["dep_id", "land_st"]]
materials = pd.read_csv("MRDS/Materials.csv")
materials = materials[["dep_id", "ore_gangue", "material"]]
model = pd.read_csv("MRDS/Model.csv")
model = model[["dep_id", "model_name"]]
physiography = pd.read_csv("MRDS/Physiography.csv")
physiography = physiography[["dep_id", "phys_div", "phys_prov", "phys_sect"]]
production_detail = pd.read_csv("MRDS/Production_detail.csv")
production_detail = production_detail[["dep_id", "yr", "code", "commod", "commod_group", "amt", "units"]]
resource_detail = pd.read_csv("MRDS/Resource_detail.csv")
resource_detail = resource_detail[["dep_id", "yr", "code", "commod", "commod_group", "import", "grd", "grd_units"]]

print(coords[coords["dep_id"].duplicated(keep=False)]["dep_id"].unique())
merged = coords.groupby("dep_id")[["wgs84_lat", "wgs84_lon"]].mean().reset_index()

tables = {
    "ages": ages,
    "commodity": commodity,
    "land_status": land_status,
    "materials": materials,
    "model": model,
    "physiography": physiography,
    "production_detail": production_detail,
    "resource_detail": resource_detail,
}

for name, dataset in tables.items():
    print(f"Grouping {name}")
    dataset = dataset.groupby("dep_id").agg(
        lambda x: ', '.join(map(str, x.unique()))
    ).reset_index()
    print(f"Merging {name}")
    merged = pd.concat([merged, dataset], axis=0)
    merged = merged.groupby("dep_id").agg(lambda x: ', '.join(map(str, x.dropna().unique()))).reset_index()
    merged.to_csv("MRDS Merged.csv", index=False)