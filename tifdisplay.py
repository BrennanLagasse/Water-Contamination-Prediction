import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# Define NLCD class values and RGB colors
nlcd_classes = [
    11, 12, 21, 22, 23, 24,
    31, 41, 42, 43, 52, 71,
    81, 82, 90, 95, 250  # Include 250 for NoData
]

nlcd_rgb = [
    (70, 107, 159),   # 11 - Open Water
    (209, 222, 248),  # 12 - Ice/Snow
    (222, 197, 197),  # 21 - Developed Open Space
    (217, 146, 130),  # 22 - Developed Low
    (235, 0, 0),      # 23 - Developed Medium
    (171, 0, 0),      # 24 - Developed High
    (179, 172, 159),  # 31 - Barren
    (104, 171, 95),   # 41 - Deciduous Forest
    (28, 95, 44),     # 42 - Evergreen Forest
    (181, 197, 143),  # 43 - Mixed Forest
    (204, 184, 121),  # 52 - Shrub/Scrub
    (223, 223, 194),  # 71 - Grassland
    (220, 217, 57),   # 81 - Pasture/Hay
    (171, 108, 40),   # 82 - Cultivated Crops
    (184, 217, 235),  # 90 - Woody Wetlands
    (108, 159, 184),  # 95 - Emergent Wetlands
    (0, 0, 0)         # 250 - NoData (black)
]

# Normalize RGB to 0-1 for matplotlib
nlcd_rgb_norm = [(r/255, g/255, b/255) for r, g, b in nlcd_rgb]

# Create colormap and normalization
cmap = ListedColormap(nlcd_rgb_norm)
norm = BoundaryNorm([v - 0.5 for v in nlcd_classes] + [nlcd_classes[-1] + 0.5], cmap.N)

# Read and plot the data
with rasterio.open("chopped_new_14.tif") as src:
    band = src.read(1)

plt.figure(figsize=(10, 6))
plt.imshow(band, cmap=cmap, norm=norm)
plt.title("NLCD-Classified GeoTIFF")
plt.axis("off")
plt.colorbar(label="Land Cover Code", ticks=nlcd_classes)
plt.show()
