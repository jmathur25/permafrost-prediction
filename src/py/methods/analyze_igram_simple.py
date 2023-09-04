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

from methods.schaefer import alt_to_surface_deformation, compute_alt_f_deformation, compute_bounding_box, plot_change
from utils import compute_stats, load_img, LatLon
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR
from data.utils import get_date_for_alos

# %%
alos1 = "ALPSRP021272170"
alos2 = "ALPSRP189022170"

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
ex = df_calm[(df_calm['date'] >= pd.to_datetime('2007')) & (df_calm['date'] <= pd.to_datetime('2010'))]
plt.plot(ex['date'], ex['alt_m'], marker='o', linestyle='-')

# %%
def try_float(x):
    try:
        return float(x)
    except:
        return np.nan
# TODO: do earlier? handle 'w'?
# TODO: fix m?
df_calm['alt_m'] = df_calm['alt_m'].apply(try_float) / 100
df_calm_d1 = df_calm[df_calm['date'] == alos_d1_nearest_calm_date]
df_calm_d2 = df_calm[df_calm['date'] == alos_d2_nearest_calm_date]

calib_point_id = 61

# %%

def get_alt_for_calib_point(calib_point_id, df):
    row1 = df[df['point_id'] == calib_point_id]
    assert len(row1) == 1

    return row1.alt_m.values[0]

def compute_deformation_for_point(point_id, df_calm_d1, df_calm_d2):
    alt1 = get_alt_for_calib_point(point_id, df_calm_d1)
    alt2 = get_alt_for_calib_point(point_id, df_calm_d2)
    
    def1 = alt_to_surface_deformation(alt1)
    def2 = alt_to_surface_deformation(alt2)
    
    # print("FRAC DEF", def1/def2)
    
    return def1 - def2

    
calib_def_12 = compute_deformation_for_point(calib_point_id, df_calm_d1, df_calm_d2)

alt1 = get_alt_for_calib_point(calib_point_id, df_calm_d1)
def1 = alt_to_surface_deformation(alt1)
alt1_hat = compute_alt_f_deformation(def1)
print(f"For an ALT of {alt1}, we got back {alt1_hat}")

alt2 = get_alt_for_calib_point(calib_point_id, df_calm_d2)
def2 = alt_to_surface_deformation(alt2)
alt2_hat = compute_alt_f_deformation(def2)
print(f"For an ALT of {alt2}, we got back {alt2_hat}")

alt2 = get_alt_for_calib_point(calib_point_id, df_calm_d2)

print(f"Estimated ground deformation at calibration point {calib_point_id}: {np.round(calib_def_12, decimals=3)} m")

# %%
intfg_unw_file = alos_isce_outputs_dir / 'interferogram/filt_topophase.unw'
intfg_unw_conncomp_file = alos_isce_outputs_dir / 'interferogram/filt_topophase.unw.conncomp'
# intfg_file = alos_isce_outputs_dir / 'interferogram/filt_topophase.flat'

# reading the multi-looked unwrapped interferogram
ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
igram_unw_phase = ds.GetRasterBand(2).ReadAsArray()
ds = None

# ds = gdal.Open(str(intfg_file), gdal.GA_ReadOnly)
# igram = ds.GetRasterBand(1).ReadAsArray()
# igram_unw_phase = np.angle(igram)# trying raw
# ds = None

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
    
bbox = compute_bounding_box(point_to_pixel[:,[1,2]])
print(bbox)

# %%
# ds = gdal.Open(str(alos_isce_outputs_dir / "geometry" / "incLocal.rdr"))
# inc_angle_img = ds.GetRasterBand(1).ReadAsArray()
# ds = None

# %%
igram_unw_phase_slice = igram_unw_phase[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]
# cc_slice = connected_components[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]

# assert np.unique(cc_slice)
plt.imshow(igram_unw_phase_slice, cmap='viridis')
plt.colorbar()
plt.title('Unwrapped phase')
plt.show()

#%%
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
    print("CP", calib_phase)
    
    los_def = calib_point_def * np.cos(incidence_angle)
    los_phase = 2 * np.pi *los_def / wavelength
    phase_diff = los_phase - calib_phase
    print(f"For calibration point: expected LOS phase: {los_phase}, actual: {calib_phase}")
    return phase_diff
    
# phase computed of alos1 - alos2, so  must be computed this way too
# incidence_angle = 38.7*np.pi/180
# wavelength = 0.2360571
with open(alos_isce_outputs_dir / "PICKLE/interferogram", "rb") as fp:
    pickle_isce_obj = pickle.load(fp)
        
wavelength = pickle_isce_obj['reference']['instrument']['radar_wavelength']
incidence_angle = pickle_isce_obj['reference']['instrument']['incidence_angle']*np.pi/180

print('radar wavelength', wavelength)
print('incidence angle', incidence_angle)

phase_corr = compute_phase_offset(
    point_to_pixel, bbox, calib_point_id, calib_def_12, igram_unw_phase_slice, incidence_angle, wavelength)

print("phase correction", phase_corr)
igram_unw_phase_slice_corr = igram_unw_phase_slice + phase_corr

# %%
def compute_deformation(igram_unw_phase_slice_corr, bbox, incidence_angle, wavelength):
    los_def = igram_unw_phase_slice_corr/(2*np.pi)*wavelength
    ground_def = los_def / np.cos(incidence_angle)
    return ground_def

igram_def = compute_deformation(igram_unw_phase_slice_corr, bbox, incidence_angle, wavelength)

# %%
plot_change(igram_def, bbox, point_to_pixel, "Predicted Deformations")

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
def construct_image(bbox, point_to_pixel, values, n=3):
    dy = bbox[1][0] - bbox[0][0]
    dx = bbox[1][1] - bbox[0][1]
    image = np.full((dy, dx), np.nan)

    # Populate the image array with ground truth deformations at specified pixel locations
    n = 3
    for (point, deformation) in zip(point_to_pixel, values):
        point_id, y, x = point
        y -= bbox[0][0]
        x -= bbox[0][1]
        n_over_2 = n // 2
        extra = n % 2
        image[y - n_over_2 : y + n_over_2 + extra , x - n_over_2 : x + n_over_2 + extra] = deformation
    return image

gt_def = construct_image(bbox, point_to_pixel, deformations)
plot_change(gt_def, bbox, point_to_pixel, "Ground-Truth Deformations")

# %%
dy = bbox[1][0] - bbox[0][0]
dx = bbox[1][1] - bbox[0][1]
alt_predictions_img = np.full((dy, dx), np.nan)
for yi in tqdm.tqdm(range(igram_def.shape[0])):
    for xi in range(igram_def.shape[1]):
        change = -igram_def[yi, xi]
        if change < 0:
            print(f"Skipping {yi}, {xi} because deformation is positive")
            continue
        alt = compute_alt_f_deformation(change)
        alt_predictions_img[yi, xi] = alt

print("Predicting ALT per point")
alt_predictions = []
for point, def_pred in zip(point_to_pixel[:,0], predicted_deformations):
    change = -def_pred
    if change < 0:
        print(f"Skipping {point} because deformation is positive")
        alt_predictions.append(np.nan)
        continue
    alt = compute_alt_f_deformation(change)
    alt_predictions.append(alt)
    
# %%
plot_change(alt_predictions_img, bbox, point_to_pixel, "ALT predictions")
# %%
gt_alt = []
for point in point_to_pixel[:,0]:
    def_12 = compute_deformation_for_point(point, df_calm_d1, df_calm_d2)
    if np.isnan(def_12):
        print(f"Point {point} has NA deformation: {point}")
        gt_alt.append(np.nan)
        continue
    change = -def_12
    if change < 0:
        print(f"Skipping point {point} because deformation is positive")
        gt_alt.append(np.nan)
        continue
    alt = compute_alt_f_deformation(change)
    gt_alt.append(alt)

gt_alt_img = construct_image(bbox, point_to_pixel, gt_alt)
plot_change(gt_alt_img, bbox, point_to_pixel, "Ground-Truth ALT")

# %%
alt_predictions = np.array(alt_predictions)
gt_alt = np.array(gt_alt)
compute_stats(alt_predictions, gt_alt)

# %%
 
