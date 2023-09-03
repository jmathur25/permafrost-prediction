# %%
%load_ext autoreload
%autoreload 2

# %%
import pickle
from isce2.components import isceobj
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

import gdal

import pandas as pd

# TODO: remove
import sys

import tqdm
sys.path.append("..")

from utils import load_img, LatLon
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR
from data.utils import get_date_for_alos

# %%
alos1 = "ALPSRP021272170"
alos2 = "ALPSRP027982170"

alos_isce_outputs_dir = ISCE2_OUTPUTS_DIR/ f"{alos1}_{alos2}"

CALM_U1_DATA_FILE = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"

# %%
lat_lon = LatLon(alos_isce_outputs_dir)
df_calm = pd.read_csv(CALM_U1_DATA_FILE, parse_dates=['date'])

# %%
# too many nans according to paper
IGNORE_POINT_IDS = [
    7,
    110,
    121
]

df_calm = df_calm[df_calm['point_id'].apply(lambda x: x not in IGNORE_POINT_IDS)]

# %%
alos_d1 = get_date_for_alos(alos1)
alos_d2 = get_date_for_alos(alos2)
print(f"Processing {alos1} on {alos_d1} and {alos2} on {alos_d2}")

# %%
def find_nearest_date(alos_date, df_calm):
    dates = df_calm['date'].drop_duplicates()
    di = abs(dates - pd.to_datetime(alos_date)).argmin()
    return dates.values[di]

alos_d1_nearest_calm_date = find_nearest_date(alos_d1, df_calm)
alos_d2_nearest_calm_date = find_nearest_date(alos_d2, df_calm)
print(f"For alos1 on {alos_d1}, closest date is {alos_d1_nearest_calm_date}")
print(f"For alos2 on {alos_d2}, closest date is {alos_d2_nearest_calm_date}")

#%%
ex = df_calm[(df_calm['point_id'] == 1) & (df_calm['date'] >= pd.to_datetime('2006')) & (df_calm['date'] < pd.to_datetime('2007'))]
plt.plot(ex['date'], ex['alt_m'], marker='o', linestyle='-')

# %%
def try_float(x):
    try:
        return float(x)
    except:
        return np.nan
# TODO: do earlier? handle 'w'?
df_calm['alt_m'] = df_calm['alt_m'].apply(try_float)
df_calm_d1 = df_calm[df_calm['date'] == alos_d1_nearest_calm_date]
df_calm_d2 = df_calm[df_calm['date'] == alos_d2_nearest_calm_date]

calib_point_id = 61

def get_alt_for_calib_point(calib_point_id, df):
    row1 = df[df['point_id'] == calib_point_id]
    assert len(row1) == 1

    return row1.alt_m.values[0]

def alt_to_surface_deformation(alt):
    # paper assumes exponential decay from 90% porosity to 45% porosity
    # in general:
    # P(z) = P_f + (Po - Pf)*e^(-kz)
    # where P(f) = final porosity
    #       P(o) = intial porosity
    #       z is rate of exponential decay
    # Without reading citation, let us assume k = 1
    # Definite integral is now: https://www.wolframalpha.com/input?i=integrate+a+%2B+be%5E%28-kz%29+dz+from+0+to+x
    po = 0.9
    pf = 0.45
    k = 1
    integral = (po * k *  alt + (pf - po) * (-np.exp(-k*alt)) + (pf - po))/k
    pw = 0.997 # g/m^3
    pi = 0.9168 # g/cm^3
    return (pw - pi) / pi * integral

def compute_deformation_for_point(point_id, df_calm_d1, df_calm_d2):
    alt1 = get_alt_for_calib_point(point_id, df_calm_d1)
    alt2 = get_alt_for_calib_point(point_id, df_calm_d2)
    
    def1 = alt_to_surface_deformation(alt1)
    def2 = alt_to_surface_deformation(alt2)
    
    return def1 - def2
    

calib_def_12 = compute_deformation_for_point(calib_point_id, df_calm_d1, df_calm_d2)

print(f"Estimated ground deformation at calibration point {calib_point_id}: {np.round(calib_def_12, decimals=3)} m")

# %%
intfg_unw_file = alos_isce_outputs_dir / 'interferogram/filt_topophase.unw'
intfg_unw_conncomp_file = alos_isce_outputs_dir / 'interferogram/filt_topophase.unw.conncomp'

# reading the multi-looked unwrapped interferogram
ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
igram_unw_phase = ds.GetRasterBand(2).ReadAsArray()
ds = None

# reading the connected component file
ds = gdal.Open(str(intfg_unw_conncomp_file), gdal.GA_ReadOnly)
connected_components = ds.GetRasterBand(1).ReadAsArray()
ds = None

# %%
df_calm_points = df_calm.drop_duplicates(subset=['point_id'])
point_to_pixel = []
for (i, row) in tqdm.tqdm(df_calm_points.iterrows(), total=len(df_calm_points)):
    point = row['point_id']
    lat = row['latitude']
    lon = row['longitude']
    px_y, px_x = lat_lon.find_closest_pixel(lat, lon)
    point_to_pixel.append([point, px_y, px_x])
point_to_pixel = np.array(point_to_pixel)
    
def compute_bounding_box(pixels, n=10):
    # Initialize min and max coordinates for y and x
    min_y = np.min(pixels[:,0])
    min_x = np.min(pixels[:,1])
    max_y = np.max(pixels[:,0])
    max_x = np.max(pixels[:,1])

    # Add 50-pixel margin to each side
    min_y = max(min_y - n, 0)
    min_x = max(min_x - n, 0)
    max_y += n
    max_x += n
    
    return ((min_y, min_x), (max_y, max_x))

bbox = compute_bounding_box(point_to_pixel[:,[1,2]])
print(bbox)

# %%
# ds = gdal.Open(str(alos_isce_outputs_dir / "geometry" / "incLocal.rdr"))
inc_angle_img = ds.GetRasterBand(1).ReadAsArray()
ds = None

# %%
igram_unw_phase_slice = igram_unw_phase[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]
# cc_slice = connected_components[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]

# assert np.unique(cc_slice)

def compute_phase_offset(
    point_to_pixel,
    bbox,
    calib_point_id,
    calib_point_def,
    igram_unw_phase_slice,
    incidence_angle,
    wavelength,
    n=3
    ):
    row = point_to_pixel[point_to_pixel[:,0] == calib_point_id]
    assert row.shape == (1, 3)
    px_y = row[0, 1] - bbox[0][0]
    px_x = row[0, 2] - bbox[0][1]
    n_over_2 = n // 2
    extra = n % 2 # if odd, we need to add 1 to in the next step
    calib_phase_slice = igram_unw_phase_slice[
        px_y - n_over_2 : px_y + n_over_2 + extra, px_x - n_over_2 : px_x + n_over_2 + extra
    ]
    calib_phase = np.mean(calib_phase_slice)
    
    los_def = calib_point_def * np.cos(incidence_angle)
    los_phase = 2 * np.pi *los_def / wavelength
    phase_diff = los_phase - calib_phase
    print(f"For calibration point: expected LOS phase: {los_phase}, actual: {calib_phase}")
    return phase_diff
    
# phase computed of alos1 - alos2, so  must be computed this way too
incidence_angle = 38.7*np.pi/180
wavelength = 0.2360571
phase_corr = compute_phase_offset(
    point_to_pixel, bbox, calib_point_id, calib_def_12, igram_unw_phase_slice, incidence_angle, wavelength)

igram_unw_phase_slice_corr = igram_unw_phase_slice + phase_corr

# %%
def compute_deformation(igram_unw_phase_slice_corr, bbox, incidence_angle, wavelength):
    los_def = igram_unw_phase_slice_corr/(2*np.pi)*wavelength
    ground_def = los_def / np.cos(incidence_angle)
    return ground_def

igram_def = compute_deformation(igram_unw_phase_slice_corr, bbox, incidence_angle, wavelength)

# %%
# Plot deformations
plt.imshow(igram_def, cmap='viridis', origin='lower')

# Add red boxes
for point in point_to_pixel:
    point_id, y, x = point
    y -= bbox[0][0]
    x -= bbox[0][1]
    plt.gca().add_patch(plt.Rectangle((x - 2.5, y - 2.5), 3, 3, fill=None, edgecolor='red', linewidth=2))
    
    # Annotate each box with the point #
    plt.annotate(f"#{point_id}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=5, color='white')

plt.colorbar()
plt.title('Deformations')
plt.show()

# %%
deformations = []
for point in point_to_pixel[:,0]:
    def_12 = compute_deformation_for_point(point, df_calm_d1, df_calm_d2)
    if np.isnan(def_12):
        print(f"Point {point} has NA deformation: {point}")
    deformations.append(def_12)
    
# %%
def get_predicted_deformations(igram_def, bbox, point_to_pixel, n=2):
    deformations = []
    for point in point_to_pixel:
        point_id, y, x = point
        y -= bbox[0][0]
        x -= bbox[0][1]
        n_over_2 = n // 2
        extra = n % 2 # if odd, we need to add 1 to in the next step
        igram_def_point = igram_def[
            y - n_over_2 : y + n_over_2 + extra, x - n_over_2 : x + n_over_2 + extra
        ]
        avg_point_def = np.mean(igram_def_point)
        deformations.append(avg_point_def)
    return deformations

predicted_deformations = get_predicted_deformations(igram_def, bbox, point_to_pixel)

# %%
r2 = r2_score(deformations, predicted_deformations)
print(f"R^2 score: {r2}")

pearson_corr, _ = pearsonr(predicted_deformations, deformations)
print(f"Pearson R: {pearson_corr}")

rmse = np.sqrt(mean_squared_error(deformations, predicted_deformations))
print(f"RMSE: {rmse}")

# %%
