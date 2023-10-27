# %%
%load_ext autoreload
%autoreload 2

# %%
import pathlib
from osgeo import gdal
import numpy as np

import sys

import pandas as pd
import tqdm
sys.path.append("/permafrost-prediction/src/py")
from methods.utils import LatLonFile
import matplotlib.pyplot as plt

# %%
d = pathlib.Path("/permafrost-prediction-shared-data/isce2_outputs/ALPSRP021272170_ALPSRP027982170_NEW/")

# %%
lat_lon = LatLonFile.RDR.create_lat_lon(d / 'geometry', full=True)

# %%
palsar = 'ALPSRP027982170'
slc = gdal.Open(str(d / f'{palsar}_slc/{palsar}.slc'), gdal.GA_ReadOnly)
assert slc.RasterCount == 1
slc_data = slc.GetRasterBand(1).ReadAsArray()

# %%
# y, x = lat_lon.find_closest_pixel(71.3061, -156.578686)

# %%
df = pd.read_csv("../../../df_alt_gt.csv")

# %%
min_y = float('inf')
max_y = float('-inf')
min_x = float('inf')
max_x = float('-inf')

for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    y, x = lat_lon.find_closest_pixel(row['latitude'], row['longitude'])
    if y < min_y:
        min_y = y
    if y > max_y:
        max_y = y
        
    if x < min_x:
        min_x = x
    if x > max_x:
        max_x = x

min_y = int(min_y)
max_y = int(max_y)
min_x = int(min_x)
max_x = int(max_x)
        
# %%
savepath = d.parent / 'jatin_ALPSRP021272170'
savepath.mkdir(exist_ok=True)

n = 100
img_slice = slc_data[min_y-n:max_y+n, min_x-n:max_x+n]
# np.save(savepath / 'raw_slc_slice.npy', img_slice)

# Define output filename
output_filename = str(savepath / 'raw_slc_slice.flat')

# Get the original dataset's geotransform
gt = slc.GetGeoTransform()

# Update geotransform for subset
new_gt = (gt[0] + (min_x-n)*gt[1], gt[1], gt[2], gt[3] + (min_y-n)*gt[5], gt[4], gt[5])

# Create a new dataset
driver = gdal.GetDriverByName('GTiff')
out_dataset = driver.Create(output_filename, xsize=img_slice.shape[1], ysize=img_slice.shape[0], bands=1,eType=gdal.GDT_CFloat32)  # CFloat32 for complex numbers

# Set the new dataset's geotransform and projection
out_dataset.SetGeoTransform(new_gt)
out_dataset.SetProjection(slc.GetProjection())

out_band = out_dataset.GetRasterBand(1)
out_band.WriteArray(img_slice)
out_dataset = None

# %%
lat_slice = lat_lon.lat_arr[min_y-n:max_y+n, min_x-n:max_x+n]
np.save(savepath / 'lat_slice.npy', lat_slice)

lon_slice = lat_lon.lon_arr[min_y-n:max_y+n, min_x-n:max_x+n]
np.save(savepath / 'lon_slice.npy', lon_slice)

# %%

src = gdal.Open('/permafrost-prediction-shared-data/isce2_outputs/ALPSRP021272170_ALPSRP027982170_NEW/interferogram/filt_topophase.flat', gdal.GA_ReadOnly)

# Create a new raster with the shape of img_slice
driver = src.GetDriver()  # Gets the same driver/format as the source
rows, cols = img_slice.shape
dst = driver.Create('test.flat', cols, rows, 1, gdal.GDT_Float32)

# Copy geotransform and projection from the source raster
dst.SetGeoTransform(src.GetGeoTransform())
dst.SetProjection(src.GetProjection())

# Write img_slice to the new raster
dst_band = dst.GetRasterBand(1)
dst_band.WriteArray(img_slice)

# Close datasets
src = None
dst = None

# %%
ds_unw = gdal.Open('/permafrost-prediction-shared-data/isce2_outputs/jatin_ALPSRP021272170/raw_slc_slice.unw', gdal.GA_ReadOnly)
data_unw = ds_unw.GetRasterBand(2).ReadAsArray()

# %%
n = 100
img_slice = slc_data[min_y-n:max_y+n, min_x-n:max_x+n]
img_angle = np.angle(img_slice)
plt.imshow(img_angle, cmap='viridis', origin='lower')
plt.colorbar()
plt.title('wrapped phase')
plt.show()

# %%
plt.imshow(data_unw, cmap='viridis', origin='lower')
plt.colorbar()
plt.title('unw phase')
plt.show()

# %%
y, x = lat_lon.find_closest_pixel(71.31072,-156.5893005)
print(y, x)

# %%
n = 5
ref_point_img = slc_data[y-n:y+n, x-n:x+n]
ref_point_angle = np.angle(ref_point_img)
plt.imshow(ref_point_angle, cmap='viridis', origin='lower')
plt.colorbar()
plt.title('wrapped phase')
plt.show()
# plt.close()

# plt.hist(ref_point_angle)
# plt.close()

# ref_point_magn = np.abs(ref_point_img)
# plt.imshow(ref_point_magn)
# plt.close()

# %%
# def adjust_dist(ref_point_angle):
#     ref_point_angle = ref_point_angle.copy()
#     cutoff = 9*np.pi/10
#     mask_upper = (ref_point_angle > cutoff)
#     mask_lower = (ref_point_angle < cutoff)
#     if np.mean(mask_upper) < np.mean(mask_lower):
#         # distribution is bottom-heavy, shift
#         ref_point_angle[mask_upper] -= 2*np.pi
#     else:
#         # opposite
#         ref_point_angle[mask_lower] += 2*np.pi
#     return ref_point_angle

# ref_point_angle_adj = adjust_dist(ref_point_angle)
# print(np.mean(ref_point_angle_adj))
# plt.hist(ref_point_angle_adj)

# %%
avg_angles = []
for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    y, x = lat_lon.find_closest_pixel(row['latitude'], row['longitude'])
    n = 5
    img = slc_data[y-n:y+n, x-n:x+n]
    angle = np.angle(img)
    # angle_adj = adjust_dist(angle)
    avg_angle = np.mean(angle)
    if avg_angle < -np.pi:
        avg_angle += 2*np.pi
    elif avg_angle > np.pi:
        avg_angle -= 2*np.pi
    assert -np.pi <= avg_angle <= np.pi
    avg_angles.append(avg_angle)
    
# %%
plt.hist(avg_angles)
    
# %%
avg_angles = np.array(avg_angles)
from scipy.stats import pearsonr
print(pearsonr(avg_angles, df['alt_m'].values))
# np.save('ALPSRP021272170_slc_inferred_relative_defs.npy', avg_angles)
    
# NOW GIVEN REF, SOLVE FOR BEST ADJUSTMENT TO EACH ANGLE SO IT IS CLOSEST TO REF?

# %%
ds = gdal.Open('/permafrost-prediction-shared-data/isce2_outputs/ALPSRP021272170_ALPSRP027982170_NEW/interferogram/filt_topophase.unw', gdal.GA_ReadOnly)
assert ds.RasterCount == 2
ds_data = ds.GetRasterBand(2).ReadAsArray()

# %%
img_slice = ds_data[min_y//8-n:max_y//8+n, min_x//4-n:max_x//4+n]
plt.imshow(img_slice)
plt.colorbar()
plt.title('unw phase')
plt.show()

# %%
# def avg_slc_val(row, n=10):
#     n_over_2 = n // 2
#     extra = n % 2
#     y, x = lat_lon.find_closest_pixel(row['latitude'], row['longitude'])
#     window = data[y-n_over_2:y+n_over_2+extra,x-n_over_2:x+n_over_2+extra]
#     assert window.shape == (n, n)
#     return np.mean(window)

# df[' '] = df.apply(avg_slc_val, axis=1)

# %%
