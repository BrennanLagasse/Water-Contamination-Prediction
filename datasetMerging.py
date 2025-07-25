#Import necessary libraries
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from geopy.distance import geodesic

#Load data from csv files
mrds = pd.read_csv("MRDS v2 Cleaned.csv")
station = pd.read_csv("GEMS Station Metadata.csv")
naics = pd.read_csv("naics_industrial_facilities.csv")

#Get location coordinates from each dataset
rock_coords = np.radians(mrds[["latitude", "longitude"]].values)
station_coords = np.radians(station[["Latitude", "Longitude"]].values)
naics_coords = np.radians(naics[["latitude83", "longitude83"]].values)

#Create first nearest neighbors model for the MRDS
knn_mrds = NearestNeighbors(n_neighbors = 5, algorithm="ball_tree", metric="haversine")
knn_mrds.fit(rock_coords)
distances_mrds, indices_mrds = knn_mrds.kneighbors(station_coords)
earth_radius_km = 6371
distances_km_mrds = distances_mrds * earth_radius_km

#Create second nearest neighbors model for NAICS
knn_naics = NearestNeighbors(n_neighbors = 5, algorithm="ball_tree", metric="haversine") #CHANGE NUMBER OF NEIGHBORS IF NEEDED
knn_naics.fit(naics_coords)
distances_naics, indices_naics = knn_naics.kneighbors(station_coords)
distances_km_naics = distances_naics * earth_radius_km

#Calculate percentage of NAICS categories within a radius
def calculate_percentage(part, whole):
    return 100 * float(part) / float(whole)

#Check if facility is within a certain radius
def is_within_radius(station_coord, facility_coord, radius_km):
    return geodesic(station_coord, facility_coord).km <= radius_km

# Define buffer radius
buffer_radius_km = 5  # 5 km radius, CHANGED IF NEEDED

# Merge datasets for each station
merged_rows = []
for i in range(len(station)):
    
    # Get station coordinates
    station_lat = station.iloc[i]["Latitude"]
    station_lon = station.iloc[i]["Longitude"]
    station_coords = (station_lat, station_lon)

    # MRDS 
    neighbor_idxs_mrds = indices_mrds[i]
    near_rows_mrds = mrds.iloc[neighbor_idxs_mrds].copy()
    near_rows_mrds["distance_km_mrds"] = distances_km_mrds[i]
    avg_mrds = near_rows_mrds.drop(columns=["latitude", "longitude"]).mean()

    # NAICS facilities within buffer
    neighbor_idxs_naics = indices_naics[i]  # Get indices of nearest NAICS facilities
    near_rows_naics = naics.iloc[neighbor_idxs_naics].copy()  # Get the NAICS rows
    near_rows_naics["distance_km_naics"] = distances_km_naics[i]  # Add distances

    facilities_within_buffer_df = near_rows_naics[near_rows_naics["distance_km_naics"] <= buffer_radius_km] #Check if less than buffer to keep

    # Calculate percentage of each NAICS category within the buffer
    category_percentages = {}
    if not facilities_within_buffer_df.empty:
        total_facilities = len(facilities_within_buffer_df)
        
        #Get percentage for each category
        for category in naics['naics_category'].unique():
            category_count = len(facilities_within_buffer_df[facilities_within_buffer_df['naics_category'] == category])
            percentage = calculate_percentage(category_count, total_facilities)
            category_percentages[f'percentage_{category}'] = percentage
    else:
        #Return zero percent if no facilities found
        for category in naics['naics_category'].unique():
            category_percentages[f'percentage_{category}'] = 0
            
    # Create row with station ID, MRDS averages, and category percentages
    merged_row = {
        "station_id": station.iloc[i]["GEMS Station Number"],
        **avg_mrds,
        **category_percentages,
    }
    merged_rows.append(merged_row)

#Add new columns to the merged data
merged_df = pd.DataFrame(merged_rows)
merged_df.to_csv("mrds_naics_merge.csv", index=False)
print(merged_df.head())