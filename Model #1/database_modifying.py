import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
gems_df = pd.read_csv("GEMS US samples.csv", sep=";")
stations_df = pd.read_csv("GEMS Station Metadata.csv")
mrds_commodity_df = pd.read_csv("MRDS Commodity.csv")
petdb_df = pd.read_csv("PetDB Rock Data.csv")
mrds_df = pd.read_csv("MRDS US Mineral Chemicals.csv")

# Clearing bad columns and rows in main dataset
#print(gems_df.columns.tolist())
gems_df.drop(["???", "Inequality"], axis=1, inplace=True)
gems_df = gems_df[~gems_df["Data Quality"].isin(["Poor", "Suspect"])]
#gems_df.to_csv("GEMS US samples.csv", index=False)
#print(gems_df.columns.tolist())

# Clearing bad columns in station metadata
#print(stations_df.columns.tolist())
stations_df.drop(stations_df.columns[1:8], axis=1, inplace=True) # Water type is all "River Station", Country is all USA
stations_df.drop(stations_df.columns[4:7], axis=1, inplace=True)
stations_df.drop(stations_df.columns[8:], axis=1, inplace=True)
stations_df.to_csv("GEMS Station Metadata.csv", index=False)
#print(stations_df.columns.tolist())

# Clearing bad columns in mrds commodity dataset
mrds_commodity_df = mrds_commodity_df[(mrds_commodity_df["country"] == "United States") & (mrds_commodity_df["score"] != "E")]
mrds_commodity_df = mrds_commodity_df[["latitude", "longitude", "com_type", "commod1", "commod2", "commod3", "oper_type", "prod_size", "dev_stat", "ore"]]
mrds_commodity_df.to_csv("MRDS Commodity.csv", index=False)
#print(mrds_commodity_df.head())

# PetDB data
important_cols = ['LATITUDE', 'LONGITUDE', 'TECTONIC SETTING', 'AGE', 'METHOD', 'ANALYZED MATERIAL', 'ROCK TYPE', 'ROCK NAME', 'SIO2', 'TIO2', 'AL2O3', 'CR2O3', 'FE2O3', 'FE2O3T', 'FEO', 'FEOT', 'NIO', 'MNO', 'MGO', 'CAO', 'SRO', 'NA2O', 'K2O', 'P2O5', 'BAO', 'LOI', 'H2O', 'H2O_M', 'H2O_P', 'SO3', 'SI', 'FE', 'MN', 'NI', 'CO', 'CU', 'CD', 'ZN', 'AS', 'AG', 'S', 'AL', 'CA', 'MG', 'CO2', 'F', 'CL', 'AU', 'B', 'BA', 'BE', 'BI', 'BR', 'CR', 'CS', 'GA', 'HF', 'K', 'LI', 'MO', 'NB', 'NI.1', 'P', 'PB', 'RB', 'S.1', 'SB', 'SC', 'SR', 'TA', 'TH', 'TI', 'U', 'V', 'Y', 'ZN.1', 'ZR', 'H2O_M.1', 'H2O_P.1', 'H2O_M.2', 'H2O_P.2']
petdb_df = petdb_df[important_cols]
petdb_df.to_csv("PetDB Rock Data.csv", index=False)
print(petdb_df.head())

# MRDS #2
mrds_df = mrds_df[["CODE_LIST", "latitude", "longitude"]]
mrds_df["CODE_LIST"] = mrds_df["CODE_LIST"].apply(
    lambda x: " ".join([chem.capitalize() + "-tot" for chem in str(x).split()])
)
mrds_df.to_csv("MRDS US Mineral Chemicals.csv", index=False)
print(mrds_df.head())

# Checking good data percentage in main dataset to gain a better overview
qualityPercent = gems_df["Data Quality"].value_counts(normalize=True)
print(qualityPercent)