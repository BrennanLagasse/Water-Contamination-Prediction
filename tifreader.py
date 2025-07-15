import rasterio
from rasterio.windows import Window
import os

input_path = "Land_Data.tif"
num_parts = 14

with rasterio.open(input_path) as src:
    part_height = src.height // num_parts
    profile = src.profile.copy()
    profile.update(height=part_height)

    for i in range(num_parts):
        y_offset = i * part_height
        current_height = part_height if i < num_parts - 1 else src.height - y_offset

        profile.update(height=current_height)

        output_path = f"chopped_new_{i+1}.tif"
        with rasterio.open(output_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                window = Window(0, y_offset, src.width, current_height)
                data = src.read(band, window=window)
                dst.write(data, band)

        print(f"Saved {output_path}")
