import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from collections import defaultdict

# Load the data
wqp = pd.read_csv("WQP Physical Chemical.csv", low_memory=False)
station = pd.read_csv("WQP Station Metadata.csv", low_memory=False)

# Merge to get HUC codes and coordinates
wqp_sample = wqp[["MonitoringLocationIdentifier", "ActivityLocation/LatitudeMeasure", 
                   "ActivityLocation/LongitudeMeasure"]].merge(
    station[["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure", "HUCEightDigitCode"]],
    on="MonitoringLocationIdentifier",
    how="left"
)

# Fill missing activity coordinates with station coordinates
wqp_sample['ActivityLocation/LatitudeMeasure'] = wqp_sample['ActivityLocation/LatitudeMeasure'].fillna(
    wqp_sample['LatitudeMeasure'])
wqp_sample['ActivityLocation/LongitudeMeasure'] = wqp_sample['ActivityLocation/LongitudeMeasure'].fillna(
    wqp_sample['LongitudeMeasure'])

# Drop rows without HUC or coordinates
wqp_sample = wqp_sample.dropna(subset=["HUCEightDigitCode", "ActivityLocation/LatitudeMeasure", 
                                        "ActivityLocation/LongitudeMeasure"])

# Extract HUC-4 region
wqp_sample['HUC4'] = wqp_sample['HUCEightDigitCode'].astype(str).str[:4]

# Keep only valid HUC-4 codes
wqp_sample = wqp_sample[wqp_sample['HUC4'].str.match(r'^\d{4}$', na=False)]

print("="*70)
print("HUC-4 BASED GEOGRAPHIC K-FOLD CROSS-VALIDATION ANALYSIS")
print("="*70)
print(f"\nTotal data points: {len(wqp_sample):,}")

huc4_counts = wqp_sample['HUC4'].value_counts().sort_index()
print(f"Number of unique HUC-4 regions: {len(huc4_counts)}")
print(f"\nHUC-4 Sample Size Statistics:")
print(f"  Mean:   {huc4_counts.mean():,.0f} samples per HUC-4")
print(f"  Median: {huc4_counts.median():,.0f} samples per HUC-4")
print(f"  Min:    {huc4_counts.min():,} samples")
print(f"  Max:    {huc4_counts.max():,} samples")
print(f"  Std:    {huc4_counts.std():,.0f}")

# Create K-fold assignments with geographic grouping
K_FOLDS = 5
print(f"\n{'='*70}")
print(f"K-FOLD STRATEGY: K={K_FOLDS} FOLDS (GEOGRAPHIC GROUPING)")
print(f"{'='*70}")

# Extract HUC-2 parent region for geographic grouping
wqp_sample['HUC2'] = wqp_sample['HUC4'].str[:2]

# Get HUC-2 regions and their sample counts
huc2_counts = wqp_sample.groupby('HUC2').size().sort_values(ascending=False)
print(f"\nHUC-2 Parent Regions found: {len(huc2_counts)}")
print(f"HUC-2 distribution:\n{huc2_counts}")

# HUC-2 Region Names
huc2_names = {
    '01': 'New England', '02': 'Mid-Atlantic', '03': 'South Atlantic-Gulf',
    '04': 'Great Lakes', '05': 'Ohio', '06': 'Tennessee',
    '07': 'Upper Mississippi', '08': 'Lower Mississippi', '09': 'Souris-Red-Rainy',
    '10': 'Missouri', '11': 'Arkansas-White-Red', '12': 'Texas-Gulf',
    '13': 'Rio Grande', '14': 'Upper Colorado', '15': 'Lower Colorado',
    '16': 'Great Basin', '17': 'Pacific Northwest', '18': 'California',
    '19': 'Alaska', '20': 'Hawaii', '21': 'Caribbean'
}

# Greedy geographic grouping: assign entire HUC-2 regions to folds
folds = [[] for _ in range(K_FOLDS)]
fold_sizes = [0] * K_FOLDS
fold_huc2s = [[] for _ in range(K_FOLDS)]

# Sort HUC-2 by size and assign to smallest fold
for huc2, count in huc2_counts.items():
    min_fold_idx = fold_sizes.index(min(fold_sizes))
    fold_huc2s[min_fold_idx].append(huc2)
    fold_sizes[min_fold_idx] += count
    
    # Add all HUC-4 regions within this HUC-2 to the fold
    huc4_in_region = wqp_sample[wqp_sample['HUC2'] == huc2]['HUC4'].unique()
    folds[min_fold_idx].extend(huc4_in_region)

# Create fold assignment dictionary
huc4_to_fold = {}
for fold_idx, huc4_list in enumerate(folds):
    for huc4 in huc4_list:
        huc4_to_fold[huc4] = fold_idx

wqp_sample['fold'] = wqp_sample['HUC4'].map(huc4_to_fold)

print(f"\nGeographic Fold Distribution:")
for i in range(K_FOLDS):
    fold_huc4s = folds[i]
    fold_size = fold_sizes[i]
    pct = (fold_size / len(wqp_sample)) * 100
    print(f"\n  Fold {i+1}:")
    print(f"    - Total samples: {fold_size:,} ({pct:.1f}%)")
    print(f"    - Number of HUC-4 subregions: {len(fold_huc4s)}")
    print(f"    - HUC-2 Parent Regions:")
    for huc2 in sorted(fold_huc2s[i]):
        huc2_name = huc2_names.get(huc2, f'Region {huc2}')
        huc2_count = len(wqp_sample[wqp_sample['HUC2'] == huc2])
        print(f"        • {huc2} ({huc2_name}): {huc2_count:,} samples")
    print(f"    - Sample HUC-4s: {', '.join(sorted(fold_huc4s)[:8])}{'...' if len(fold_huc4s) > 8 else ''}")

# Use all data points for map
wqp_plot = wqp_sample
print(f"\nVisualizing all {len(wqp_plot):,} data points...")

# Create the map - color by fold assignment
fig = plt.figure(figsize=(14, 10))
ax1 = plt.axes(projection=ccrs.LambertConformal())

# Define map extent for Continental US
extent = [-125, -66, 24, 50]

# Colors for folds
fold_colors = plt.cm.Set1(np.linspace(0, 1, K_FOLDS))

# Create map
ax1.set_extent(extent, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, edgecolor='gray', alpha=0.5)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
ax1.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)

for fold_idx in range(K_FOLDS):
    fold_data = wqp_plot[wqp_plot['fold'] == fold_idx]
    ax1.scatter(
        fold_data['ActivityLocation/LongitudeMeasure'],
        fold_data['ActivityLocation/LatitudeMeasure'],
        c=[fold_colors[fold_idx]],
        s=2,
        alpha=0.4,
        label=f'Fold {fold_idx+1} (n={fold_sizes[fold_idx]:,})',
        transform=ccrs.PlateCarree(),
        edgecolors='none',
        rasterized=True
    )

ax1.set_title(f'K-Fold Geographic Assignment (K={K_FOLDS})\nBased on HUC-4 Regions', 
              fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=10)

plt.tight_layout()
print("Saving map (this may take 30-60 seconds with all data points)...")
plt.savefig('huc4_kfold_map.png', dpi=250, bbox_inches='tight')
plt.show()

# Save fold assignments for use in training
fold_assignment = pd.DataFrame({
    'HUC4': list(huc4_to_fold.keys()),
    'fold': list(huc4_to_fold.values())
})
fold_assignment.to_csv('huc4_fold_assignments.csv', index=False)

print(f"\nFiles saved:")
print(f"  - huc4_kfold_map.png (visualization)")
print(f"  - huc4_fold_assignments.csv (fold assignments for training)")
print("="*70)