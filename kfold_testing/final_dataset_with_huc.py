import pandas as pd
import numpy as np

print("Adding HUC4 to final dataset (from HUC_Code.csv)")

# Load final dataset
print("\nLoading final_dataset.csv")
final_df = pd.read_csv("final_dataset.csv")
print(f"Final dataset shape: {final_df.shape}")

# Load HUC codes
print("\nLoading HUC_Code.csv")
huc_df = pd.read_csv("HUC_Code.csv")
print(f"HUC_Code shape: {huc_df.shape}")

# Add HUCEightDigitCode to final_df
final_with_huc = final_df.copy()
final_with_huc['HUCEightDigitCode'] = huc_df['HUCEightDigitCode'].values

# Extract HUC4
print("\nExtracting HUC4 from HUCEightDigitCode...")
final_with_huc['HUC4'] = final_with_huc['HUCEightDigitCode'].astype(str).str[:4]

# Replace 'nan' string with actual NaN
final_with_huc['HUC4'] = final_with_huc['HUC4'].replace('nan', pd.NA)

# Drop the full 8-digit code
final_with_huc = final_with_huc.drop('HUCEightDigitCode', axis=1)

# Statistics
matched = final_with_huc['HUC4'].notna().sum()
missing = final_with_huc['HUC4'].isna().sum()

print(f"\n{'='*70}")
print("RESULTS:")
print(f"{'='*70}")
print(f"Total samples in final dataset: {len(final_df):,}")
print(f"Samples with HUC4:              {matched:,} ({matched/len(final_df)*100:.2f}%)")
print(f"Samples missing HUC4:           {missing:,} ({missing/len(final_df)*100:.2f}%)")

if matched > 0:
    print(f"\nUnique HUC4 regions: {final_with_huc['HUC4'].nunique()}")
    print(f"\nTop 10 HUC4 regions by sample count:")
    print(final_with_huc['HUC4'].value_counts().head(10))

# Check Arsenic-specific statistics
if 'CharacteristicName' in final_with_huc.columns:
    arsenic_mask = final_with_huc['CharacteristicName'] == 'Arsenic'
    total_arsenic = arsenic_mask.sum()
    arsenic_with_huc = (arsenic_mask & final_with_huc['HUC4'].notna()).sum()
    arsenic_missing = total_arsenic - arsenic_with_huc
    
    print(f"\n{'='*70}")
    print(f"ARSENIC-SPECIFIC STATISTICS:")
    print(f"{'='*70}")
    print(f"Total Arsenic samples:        {total_arsenic:,}")
    print(f"Arsenic with HUC4:            {arsenic_with_huc:,} ({arsenic_with_huc/total_arsenic*100:.2f}%)")
    print(f"Arsenic missing HUC4:         {arsenic_missing:,} ({arsenic_missing/total_arsenic*100:.2f}%)")

# Save the result
output_file = "final_dataset_with_huc.csv"
final_with_huc.to_csv(output_file, index=False)

print(f"Saved to: {output_file}")
print(f"  Columns: {len(final_with_huc.columns)} (original: {len(final_df.columns)})")
print(f"  New column: HUC4")