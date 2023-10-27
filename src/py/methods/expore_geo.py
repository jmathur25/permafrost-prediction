# %%
%load_ext autoreload
%autoreload 2
# %%
import numpy as np
from osgeo import gdal
import pathlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import pandas as pd
from py.methods.run_mintpy_ts_analysis import compute_bounding_box, plot_change

from utils import LatLon, LatLonGeo

# %%
isce_output_dir = pathlib.Path("../data/isce2_outputs/ALPSRP021272170_ALPSRP027982170/")
intfg_unw_file_geo = isce_output_dir / "interferogram/filt_topophase.unw.geo"
ds = gdal.Open(str(intfg_unw_file_geo), gdal.GA_ReadOnly)
igram_unw_phase_geo = ds.GetRasterBand(2).ReadAsArray()

intfg_unw_file = isce_output_dir / "interferogram/filt_topophase.unw"
ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
igram_unw_phase = ds.GetRasterBand(2).ReadAsArray()

# %%
plt.imshow(igram_unw_phase)

# %%
plt.imshow(igram_unw_phase_geo)

# %%
lat_lon_geo = LatLonGeo(intfg_unw_file_geo)
lat_lon = LatLon(isce_output_dir / "geometry")

# %%
df_alt_gt = pd.read_csv("/permafrost-prediction/alt_gt.csv")
df_alt_gt.set_index("point_id", inplace=True)

# %%
point_to_pixel_geo = []
point_to_pixel = []
point_to_pixel_geo_2 = []
for point, row in df_alt_gt.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]

    y, x = lat_lon_geo.find_closest_pixel(lat, lon)
    point_to_pixel_geo.append([point, y, x])

    y, x = lat_lon.find_closest_pixel(lat, lon)
    point_to_pixel.append([point, y, x])
    
    y, x = src.index(lon, lat)
    point_to_pixel_geo_2.append([point, y, x])
point_to_pixel_geo = np.array(point_to_pixel_geo)
point_to_pixel = np.array(point_to_pixel)
point_to_pixel_geo_2 = np.array(point_to_pixel_geo_2)

# %%
bbox_geo = compute_bounding_box(point_to_pixel_geo[:, [1, 2]])
bbox = compute_bounding_box(point_to_pixel[:, [1, 2]])

# %%
igram_unw_phase_geo_slice = -igram_unw_phase_geo[bbox_geo[0][0] : bbox_geo[1][0], bbox_geo[0][1] : bbox_geo[1][1]]
igram_unw_phase_slice = -igram_unw_phase[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]

# %%
plot_change(igram_unw_phase_slice, bbox, point_to_pixel, "Radar-Coord Phase")
plot_change(igram_unw_phase_geo_slice, bbox_geo, point_to_pixel_geo, "Geo-Coord Phase")

# %%
vmin = min(igram_unw_phase_geo_slice.min(), igram_unw_phase_slice.min())
vmax = max(igram_unw_phase_geo_slice.max(), igram_unw_phase_slice.max())

# Create a figure and axis
fig, ax = plt.subplots(1, 2)  # 1 row, 2 columns

# Display images
cax1 = ax[0].imshow(igram_unw_phase_geo_slice, cmap="viridis", vmin=vmin, vmax=vmax)
cax2 = ax[1].imshow(igram_unw_phase_slice, cmap="viridis", vmin=vmin, vmax=vmax)

# Turn off axis numbering/ticks for both subplots
ax[0].axis("off")
ax[1].axis("off")

cbar = fig.colorbar(cax1, ax=ax.ravel().tolist(), orientation="vertical")

plt.show()

# %%
import rasterio


src = rasterio.open("/permafrost-prediction/src/py/filt_topophase.unw.geo.tif")

# %%
array = src.read()

# %%
