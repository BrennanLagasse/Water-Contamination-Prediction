import pandas as pd
pd.set_option("display.max_columns", None)
import os
print("Saving to:", os.getcwd())

# Column selection first based on usefulness, then further columns removed based on % missing and duplicates
ages = pd.read_csv("MRDS/Ages.csv") # Columns 80% empty and removed after
ages = ages[["dep_id", "age_tp", "age_young_cd", "age_young"]]
commodity = pd.read_csv("MRDS/Commodity.csv")
commodity = commodity[["dep_id", "commod_tp", "commod_group", "import"]]
coords = pd.read_csv("MRDS/Coords.csv")
coords = coords[["dep_id", "wgs84_lat", "wgs84_lon"]]
land_status = pd.read_csv("MRDS/Land_status.csv") # Columns 33% empty and removed after
land_status = land_status[["dep_id", "land_st"]]
materials = pd.read_csv("MRDS/Materials.csv") # Columns 72% empty and removed after
materials = materials[["dep_id", "ore_gangue", "material"]]
model = pd.read_csv("MRDS/Model.csv") # Columns 94% empty and removed after
model = model[["dep_id", "model_name"]]
physiography = pd.read_csv("MRDS/Physiography.csv") # phys_sect 33% empty and removed after
physiography = physiography[["dep_id", "phys_div", "phys_prov", "phys_sect"]]
production_detail = pd.read_csv("MRDS/Production_detail.csv") # Columns 97-99% empty and removed after
production_detail = production_detail[["dep_id", "yr", "commod_group", "amt", "units"]]
resource_detail = pd.read_csv("MRDS/Resource_detail.csv") # Columns 97% empty and removed after
resource_detail = resource_detail[["dep_id", "yr", "commod_group", "import", "grd", "grd_units"]]

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
    merged = pd.merge(merged, dataset, on="dep_id", how="outer")
    merged = merged.groupby("dep_id").agg(lambda x: ', '.join(map(str, x.dropna().unique()))).reset_index()
    merged.to_csv("MRDS Merged.csv", index=False)