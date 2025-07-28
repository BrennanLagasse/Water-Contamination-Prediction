import pandas as pd

stations_df = pd.read_csv("GEMS Station Metadata.csv")
gems_df = pd.read_csv("GEMS US samples.csv")

# Creating the testing data
testing_data = gems_df.merge(stations_df[["GEMS Station Number", "Latitude", "Longitude"]], left_on="Station", right_on="GEMS Station Number", how="left")
testing_data.drop(["Station", "GEMS Station Number", "Data Quality"], inplace=True, axis=1)
print(testing_data.head())
testing_data.to_csv("Testing Data.csv", index=False)