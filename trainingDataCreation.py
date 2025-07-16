import pandas as pd
from sklearn.preprocessing import LabelEncoder
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
stations_df = pd.read_csv("MRDS + Station Merge.csv")
gems_df = pd.read_csv("GEMS US samples.csv")

# Creating the testing data
bad_chemicals = ["As-Tot", "Pb-Tot", "Cd-Tot", "Hg-Tot", "Cr-Tot", "Ni-Tot", "Cu-Tot", "Zn-Tot", "Se-Tot", "Ba-Tot", "B-Tot", "CN-Tot", "NO3N", "NO2N", "NH3N", "NH4N", "TON", "TKN", "DON", "DKN", "TP", "TDP", "TRP", "TPP", "SO4-Tot", "Cl-Tot", "EC", "TDS", "TS", "Fe-Tot", "Mn-Tot", "Al-Tot", "POC", "TOC", "BOD", "COD", "TOTCOLI", "FECALCOLI", "FECALSTREP", "DIELDRIN", "ENDRIN", "ALDRIN", "MIREX", "BHC-gamma", "BENZENE"]
testing_data = gems_df.merge(stations_df, left_on="Station", right_on="GEMS Station Number", how="left")
testing_data.drop(["Station", "GEMS Station Number", "Data Quality", "Method", "Time"], inplace=True, axis=1) # Time shouldn't matter too much
testing_data = testing_data[testing_data["Chemical"].isin(bad_chemicals)] # Only selecting rows worth predicting for
testing_data = testing_data[testing_data["Unit"] == "mg/l"] # Only predicting rows with mg/l unit
testing_data.drop("Unit", inplace=True, axis=1)
ground_truth = testing_data["Amount"]
testing_data.drop("Amount", inplace=True, axis=1)

# Encoding time and chemical
testing_data["Date"] = pd.to_datetime(testing_data["Date"])
testing_data["Year"] = testing_data["Date"].dt.year
testing_data["Month"] = testing_data["Date"].dt.month
testing_data["Day"] = testing_data["Date"].dt.day
testing_data.drop("Date", inplace=True, axis=1)
testing_data["ChemicalEncoded"] = LabelEncoder().fit_transform(testing_data["Chemical"])
testing_data["BasinEncoded"] = LabelEncoder().fit_transform(testing_data["Main Basin"])
testing_data.drop(["Chemical", "Main Basin"], inplace=True, axis=1)
print(testing_data.head())

# Checking for nans
print(testing_data.isna().mean() * 100)
testing_data = testing_data.dropna()
print(testing_data.isna().mean() * 100)
ground_truth.to_csv("Testing Truth.csv", index=False)
testing_data.to_csv("Testing Data.csv", index=False)