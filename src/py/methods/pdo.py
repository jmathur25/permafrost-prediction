# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from osgeo import gdal
import zipfile
import xml.etree.ElementTree as ET

from scipy.stats import pearsonr
import sys

import pandas as pd
import tqdm


sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import LiuSMM
from data.consts import CALM_PROCESSSED_DATA_DIR
from methods.run_analysis import compute_deformation, plot_change
from methods.utils import LatLonFunc, compute_bounding_box, load_calm_data

# %%
uavsar_barrow_folder = pathlib.Path('/permafrost-prediction-shared-data/uavsar/barrow/')

int_barrow_path = uavsar_barrow_folder / 'barrow_15018_17069-003_17098-001_0087d_s01_L090HH_02.int.grd'

# %%
im = np.fromfile(int_barrow_path, np.complex64)
nrows = 22567
ncols = 22235
im = im.reshape(nrows, ncols)
phase = np.angle(im)
plt.imshow(phase)

# %%
top_left_lat = 71.452938000000017
top_left_lon = -156.807654130000003
lat_spacing = -0.0000555600000000
lon_spacing = 0.0001111100000000
lat_lines, lon_lines = 22567, 22235
assert (lat_lines, lon_lines) == im.shape
bottom_right_lat = top_left_lat + lat_lines * lat_spacing
bottom_right_lon = top_left_lon + lon_lines * lon_spacing
lat_lon = LatLonFunc(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, lat_spacing, lon_spacing)


# %%
print(lat_lon.find_closest_pixel(top_left_lat, top_left_lon))
print(lat_lon.find_closest_pixel(bottom_right_lat, bottom_right_lon))

# %%
chunk_size=(5,7)

# %%
def chunk_and_average(data, chunk_size):
    y_num_blocks = data.shape[0] // chunk_size[0]
    x_num_blocks = data.shape[1] // chunk_size[1]
    if y_num_blocks * chunk_size[0] != data.shape[0]:
        y_num_blocks += 1
    if x_num_blocks * chunk_size[1] != data.shape[1]:
        x_num_blocks += 1
    
    # Determine the shape of the output matrix based on chunk size
    out_shape = (y_num_blocks, x_num_blocks)
    
    # Create an output matrix of zeros with the shape of out_shape
    averaged_data = np.zeros(out_shape, dtype=complex)
    
    # Iterate over the chunks and compute the average
    for i in tqdm.tqdm(range(y_num_blocks)):
        for j in range(x_num_blocks):
            start_y = i * chunk_size[0]
            start_x = j * chunk_size[1]
            chunk = data[start_y:start_y+chunk_size[0], start_x:start_x+chunk_size[1]]
            assert 0 < chunk.shape[0] <= chunk_size[0]
            assert 0 < chunk.shape[1] <= chunk_size[1]
            m = np.mean(chunk)
            assert not np.isnan(m).any(), f"chunk: {chunk}"
            averaged_data[i, j] = m
    return averaged_data

im_chunk = chunk_and_average(im, chunk_size)

# %%
lat_lon_chunk = LatLonFunc(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon, lat_spacing, lon_spacing, chunk_size=chunk_size)


# %%
calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
df_calm = load_calm_data(calm_file, [], 2017, 2017)
df_calm_points = df_calm[['point_id', 'latitude', 'longitude']].drop_duplicates()
df_calm_points['point_id'] = df_calm_points['point_id'].astype(int)
df_calm_points = df_calm_points.set_index('point_id')

# %%
im_chunk_angles = np.angle(im_chunk)
point_to_pixel = []
wrapped_phase_changes = []
jatin_unw_phase_changes = []
for point_id, row in df_calm_points.iterrows():
    y, x = lat_lon_chunk.find_closest_pixel(row['latitude'], row['longitude'])
    point_to_pixel.append([point_id, y, x])
    phase = im_chunk_angles[y, x]
    wrapped_phase_changes.append(phase)
    if phase < -np.pi + 0.5:
        phase += 2*np.pi
    jatin_unw_phase_changes.append(phase)

wrapped_phase_changes = np.array(wrapped_phase_changes)
jatin_unw_phase_changes = np.array(jatin_unw_phase_changes)
point_to_pixel = np.array(point_to_pixel)

# %%
plt.hist(wrapped_phase_changes)
plt.title("Wrapped Phase Changes")

# %%
plt.hist(jatin_unw_phase_changes)  
plt.title("Jatin Unw Phase Changes")

# %%
bbox = compute_bounding_box(point_to_pixel[:, [1,2]], n=30)

# %%
# y, x = 2346, 8
# n_vert = 2000
# n_horiz = 1000
# plt.imshow(phase[y - n_vert:y + n_vert,x - n_horiz:x + n_horiz])
im_phase = np.angle(im_chunk)
phase_slice = im_phase[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]
plot_change(phase_slice, bbox, point_to_pixel, 'Wrapped Phase')

# %%
y, x = lat_lon.find_closest_pixel(row['latitude'], row['longitude'])
y_chunk, x_chunk = lat_lon_chunk.find_closest_pixel(row['latitude'], row['longitude'])
assert y_chunk == y//chunk_size[0]
assert x_chunk == x//chunk_size[1]


# %%
snaphu_input_file = uavsar_barrow_folder / "processed" / "hh.flat"
if snaphu_input_file.exists():
    os.remove(snaphu_input_file)
snaphu_input_file.parent.mkdir(exist_ok=True)

src = gdal.Open('/permafrost-prediction-shared-data/isce2_outputs/ALPSRP021272170_ALPSRP027982170_NEW/interferogram/filt_topophase.flat', gdal.GA_ReadOnly)

# Create a new raster with the shape of img_slice
driver = src.GetDriver()  # Gets the same driver/format as the source
rows, cols = im_chunk.shape
dst = driver.Create(str(snaphu_input_file), cols, rows, 1, gdal.GDT_CFloat32)

# Copy geotransform and projection from the source raster
# dst.SetGeoTransform(src.GetGeoTransform())
# dst.SetProjection(src.GetProjection())

# Write img_slice to the new raster
dst_band = dst.GetRasterBand(1)
dst_band.WriteArray(im_chunk)

# Close datasets
src = None
dst = None

# %%
# ds = gdal.Open(str(snaphu_input_file))
# tmp = ds.GetRasterBand(1).ReadAsArray()

# TODO: make test_snaphu.py a callable func
snaphu_output_file = uavsar_barrow_folder / "processed" / "hh.unw"
ds = gdal.Open(str(snaphu_output_file))
im_chunk_unw = ds.GetRasterBand(1).ReadAsArray()
ds = None

# %%
phase = np.angle(im_chunk)
phase_slice = phase[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]
plot_change(phase_slice, bbox, point_to_pixel, 'Wrapped Phase')

# %%
phase_slice_unw = im_chunk_unw[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]
plot_change(phase_slice_unw, bbox, point_to_pixel, 'Unwrapped Phase')

# %%
df_alts = df_calm.set_index('point_id')
phases = []
for (point_id, px_y, px_x) in point_to_pixel:
    phase = im_chunk_unw[px_y, px_x]
    phases.append(phase)
phases = np.array(phases)

# %%
alts = df_alts['alt_m'].values
not_nan_mask = ~np.isnan(alts)
alts_subset = alts[not_nan_mask]
phases_subset = wrapped_phase_changes[not_nan_mask]
print(pearsonr(alts_subset, phases_subset))

plt.scatter(phases, alts)
plt.xlabel("Phases")
plt.ylabel("ALTs")

# %%
avg_look_angle_deg_near_range = 21.8
avg_look_angle_deg_far_range = 65.35

# TODO: instead of linear interp, use plane height
# TODO: valid only for small plots. Do for all?
_, avg_x = point_to_pixel[:,[1,2]].mean(axis=0)

frac_range = avg_x/im_chunk_unw.shape[1]
slope= (avg_look_angle_deg_far_range-avg_look_angle_deg_near_range)/im_chunk_unw.shape[1]

avg_inc_angle_deg = slope*avg_x + avg_look_angle_deg_near_range
print("avg inc angle (deg):", avg_inc_angle_deg)

avg_inc_angle = avg_inc_angle_deg*np.pi/180
radar_wavelength = 0.238403545

im_deformation = compute_deformation(im_chunk_unw, avg_inc_angle, radar_wavelength)
jatin_deformations = compute_deformation(jatin_unw_phase_changes, avg_inc_angle, radar_wavelength)

# %%
# CALIBRATE
# deformations = []
# for (point_id, px_y, px_x) in point_to_pixel:
#     deformation = im_deformation[px_y, px_x]
#     deformations.append(deformation)
# deformations = np.array(deformations)

smm = LiuSMM()
gt_deformations = np.array([smm.deformation_from_alt(alt) for alt in alts])

def_95 = np.percentile(jatin_deformations, 95)
def_5 = np.percentile(jatin_deformations, 5)
gt_def_95 = np.percentile(gt_deformations, 95)
gt_def_5 = np.percentile(gt_deformations, 5)

lhs = (jatin_deformations - np.median(jatin_deformations))/(def_95-def_5)

alpha = 1.0355
deformations_cal = lhs*(gt_def_95-gt_def_5) + np.median(gt_def_95)

# %%
deformations = []
for (point_id, px_y, px_x) in point_to_pixel:
    # px_y -= bbox[0][0]
    # px_x -= bbox[0][1]
    deformation = im_deformation[px_y, px_x]
    deformations.append(deformation)
deformations = np.array(deformations)

# %%

deformations_adj = 1.0355*(0.386*deformations + 1.85)

alts = df_alts['alt_m'].values
not_nan_mask = ~np.isnan(alts)
alts_subset = alts[not_nan_mask]
deformation_subset = deformations_adj[not_nan_mask]
print(pearsonr(alts_subset, deformation_subset))

plt.scatter(deformations_adj, alts)
plt.xlabel("Deformation")
plt.ylabel("ALTs")

# %%
weights = np.ones_like(deformations_cal) / len(deformations_cal)
plt.hist(deformations_cal, weights=weights, color='b', alpha=0.5)
plt.hist(gt_deformations, weights=weights, color='r', bins=20, alpha=0.5)

# %%
