import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic

arsenic = pd.read_csv('arsenics_real.csv')
station_coords = pd.read_csv('trainingfinal.csv')
naics = pd.read_csv('naics_industrial_facilities.csv')
wqp_mrds = pd.read_csv('WQP + MRDS.csv')

# Combine arsenic data with station coordinates
arsenic_stations = pd.concat([arsenic, station_coords], axis=1)

# Ensure the required columns for the first merge are available.
keep_arsenic_cols = [
    'chem_ResultMeasureValue',
    'LatitudeMeasure',
    'LongitudeMeasure',
]
arsenic_stations = arsenic_stations[keep_arsenic_cols]

# Filter WQP + MRDS to only include rows with Arsenic
wqp_mrds = wqp_mrds[wqp_mrds['CharacteristicName'] == 'Arsenic'].copy()

# Uses NearestNeighbors for spatial join
# Get the coordinates for each dataframe
coords_arsenic = arsenic_stations[['LatitudeMeasure', 'LongitudeMeasure']].values
coords_wqp = wqp_mrds[['ActivityLocation/LatitudeMeasure', 'ActivityLocation/LongitudeMeasure']].values

# Change coordinates to be in radians
coords_wqp_radians = np.radians(coords_wqp)
coords_arsenic_radians = np.radians(coords_arsenic)

# Define the distance threshold for the first merge
distance_threshold_km = 0.1  # 100 meters
earth_radius_km = 6371  # Earth's radius for haversine conversion

# Fit NearestNeighbors to the WQP + MRDS data !!
knn_wqp = NearestNeighbors(n_neighbors=1, algorithm = 'ball_tree', metric='haversine')
knn_wqp.fit(coords_wqp_radians)

# Find all neighbors within the specified radius for each arsenic point
distances_rad, indices = knn_wqp.kneighbors(coords_arsenic_radians, return_distance=True)
distances_km = distances_rad * earth_radius_km

# Create a list of dictionaries for the final DataFrame
matched_data = []
for i in range(len(arsenic_stations)):
    
    # Check if the closest match is within the distance threshold
    if distances_km[i][0] <= distance_threshold_km:
        # Get the index of the single closest match
        wqp_idx = indices[i][0]
        arsenic_row = arsenic_stations.iloc[i]
        wqp_row = wqp_mrds.iloc[wqp_idx]

        # Construct the matched row
        matched_row = {
            'ArsenicMeasureValue': arsenic_row['chem_ResultMeasureValue'],
            'LatitudeMeasure': arsenic_row['LatitudeMeasure'],
            'LongitudeMeasure': arsenic_row['LongitudeMeasure'],
            # Add all relevant columns from the wqp_mrds row
            'WellDepth': wqp_row['WellDepthMeasure/MeasureValue'],
            'mrds_commod_tp_emb_0': wqp_row['mrds_commod_tp_emb_0'],
            'mrds_commod_tp_emb_1': wqp_row['mrds_commod_tp_emb_1'],
            'mrds_commod_tp_emb_2': wqp_row['mrds_commod_tp_emb_2'],
            'mrds_commod_tp_emb_3': wqp_row['mrds_commod_tp_emb_3'],
            'mrds_commod_group_x_emb_0': wqp_row['mrds_commod_group_x_emb_0'],
            'mrds_commod_group_x_emb_1': wqp_row['mrds_commod_group_x_emb_1'],
            'mrds_commod_group_x_emb_2': wqp_row['mrds_commod_group_x_emb_2'],
            'mrds_commod_group_x_emb_3': wqp_row['mrds_commod_group_x_emb_3'],
            'mrds_import_x_emb_0': wqp_row['mrds_import_x_emb_0'],
            'mrds_import_x_emb_1': wqp_row['mrds_import_x_emb_1'],
            'mrds_phys_div_emb_0': wqp_row['mrds_phys_div_emb_0'],
            'mrds_phys_div_emb_1': wqp_row['mrds_phys_div_emb_1'],
            'mrds_phys_div_emb_2': wqp_row['mrds_phys_div_emb_2'],
            'mrds_phys_div_emb_3': wqp_row['mrds_phys_div_emb_3'],
            'mrds_phys_prov_emb_0': wqp_row['mrds_phys_prov_emb_0'],
            'mrds_phys_prov_emb_1': wqp_row['mrds_phys_prov_emb_1'],
            'mrds_phys_prov_emb_2': wqp_row['mrds_phys_prov_emb_2'],
            'mrds_phys_prov_emb_3': wqp_row['mrds_phys_prov_emb_3'],
            'mrds_phys_prov_emb_4': wqp_row['mrds_phys_prov_emb_4'],
            'mrds_phys_prov_emb_5': wqp_row['mrds_phys_prov_emb_5'],
        }
        matched_data.append(matched_row)

# The result of the first merge is now stored in 'merged'
if matched_data:
    merged = pd.DataFrame(matched_data)
    print("First merge complete. Merged a total of {} rows.".format(len(merged)))
else:
    print("First merge found no matches. Exiting before NAICS analysis.")
    exit()

# Uses merged dataframe to find NAICS facilities within a specified buffer and calculate the percentage of each category.
# Get location coordinates from the NAICS dataset
naics_coords = np.radians(naics[["latitude83", "longitude83"]].values)
unique_naics_categories = naics['naics_category'].unique()

# Create nearest neighbors model for NAICS using Ball Tree for efficiency
knn_naics = NearestNeighbors(n_neighbors=len(naics), algorithm="ball_tree", metric="haversine")
knn_naics.fit(naics_coords)

# Find all NAICS facilities within a specified radius for each merged point
buffer_radius_km = 10  # 10 km radius
buffer_radius_radians = buffer_radius_km / earth_radius_km

distances_naics, indices_naics = knn_naics.radius_neighbors(np.radians(merged[["LatitudeMeasure", "LongitudeMeasure"]].values), radius=buffer_radius_radians, return_distance=True)

# Get percentages for NAICS categories
merged_rows_naics = []
for i in range(len(merged)):
    # Create a dictionary to hold category percentages for the current row
    category_percentages = {}

    # Get indices of nearest NAICS facilities within the buffer
    neighbor_idxs_naics = indices_naics[i]

    if len(neighbor_idxs_naics) > 0:
        # Get the NAICS rows corresponding to the neighbors
        facilities_within_buffer_df = naics.iloc[neighbor_idxs_naics].copy()
        total_facilities = len(facilities_within_buffer_df)

        # Get percentage for each category
        for category in unique_naics_categories:
            category_count = len(facilities_within_buffer_df[facilities_within_buffer_df['naics_category'] == category])
            percentage = (category_count / total_facilities) * 100
            category_percentages[f'percentage_{category}'] = percentage
    else:
        # Return zero percent if no facilities found
        for category in unique_naics_categories:
            category_percentages[f'percentage_{category}'] = 0
            
    # Create a new row with the percentage columns
    merged_row = {
        **category_percentages,
    }
    merged_rows_naics.append(merged_row)

# Create a DataFrame from the calculated percentages
naics_percentages_df = pd.DataFrame(merged_rows_naics)

# Add the new percentage columns to the merged_ca dataframe
merged_final_df = pd.concat([merged.reset_index(drop=True), naics_percentages_df], axis=1)

# Save the final merged dataframe to a new CSV file
merged_final_df.to_csv("wqp_mrds_naics_merge.csv", index=False)

print("\nSuccessfully created 'wqp_mrds_naics_merge.csv'")
print("\nHead of the final merged dataframe:")
print(merged_final_df.head())