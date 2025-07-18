from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
mrds = pd.read_csv("MRDS v2 Cleaned.csv")
station = pd.read_csv("GEMS Station Metadata.csv")

rock_coords = np.radians(mrds[["latitude", "longitude"]].values)
station_coords = np.radians(station[["Latitude", "Longitude"]].values)

knn = NearestNeighbors(n_neighbors = 5, algorithm="ball_tree", metric="haversine")
knn.fit(rock_coords)

distances, indices = knn.kneighbors(station_coords)
earth_radius_km = 6371
distances_km = distances * earth_radius_km

# Example: for each chem point, take the mean of its top 10 neighbors’ data
merged_rows = []
for i, neighbor_idxs in enumerate(indices):
    near_rows = mrds.iloc[neighbor_idxs].copy()
    near_rows["distance_km"] = distances_km[i]

    # Average neighbors (excluding coordinates)
    avg = near_rows.drop(columns=["latitude", "longitude"]).sum()
    
    # Combine with chem row
    combined = pd.concat([station.iloc[i], avg])
    merged_rows.append(combined)

merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv("MRDS + Station Merge.csv", index=False)
print(merged_df.head())