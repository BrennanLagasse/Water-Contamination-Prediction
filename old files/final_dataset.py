import pandas as pd

# Replace 'your_column_name' with the actual name of the column containing the desired column names
column_names_df = pd.read_csv('WQP_MRDS_gNATSGO.csv')
desired_columns = column_names_df['Feature'].tolist()

print("Desired columns:")
print(desired_columns)

# Replace 'your_cleaned_file.csv' with the actual name of your cleaned data file
cleaned_df = pd.read_csv('(Cleaned) WQP + MRDS + gNATSGO.csv')

# Filter the DataFrame to keep only the desired columns
filtered_df = cleaned_df[desired_columns]

print("\nFiltered DataFrame head:")
#display(filtered_df.head())

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('WQP_MRDS_gNATSGO.csv', index=False)

print("\nFiltered data saved to 'WQP_MRDS_gNATSGO.csv'")