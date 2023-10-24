
from datetime import date, datetime
import sys

import numpy as np
import pandas as pd


sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import alt_to_surface_deformation, compute_alt_f_deformation
from methods.schaefer import process_scene_pair
from data.utils import get_date_for_alos
from methods.utils import compute_stats, prepare_calm_data, prepare_temp
from methods.igrams import JATIN_SINGLE_SEASON_2006_IGRAMS
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR

igrams_usable = JATIN_SINGLE_SEASON_2006_IGRAMS = [
    ('ALPSRP027982170', 'ALPSRP026522180'),
    ('ALPSRP026522180', 'ALPSRP021272170'),
    # ('ALPSRP021272170', 'ALPSRP020901420'),
    ('ALPSRP020901420', 'ALPSRP019812180'),
    # ('ALPSRP019812180', 'ALPSRP017332180'),
    # ('ALPSRP017332180', 'ALPSRP016671420'),
]
print("OVERRIDE IGRAMS")
igrams_usable = igrams_usable[0:1]

# igrams_usable = []
# for (alos2, alos1) in JATIN_SINGLE_SEASON_2006_IGRAMS:
#     d = ISCE2_OUTPUTS_DIR / f"{alos2}_{alos1}"
#     if not (d / 'interferogram/filt_topophase.unw').exists():
#         print("TODO:", d)
#     else:
#         igrams_usable.append((alos2, alos1))
        
calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

paper_specified_ignore = [7, 110, 121]
data_specified_ignore = [21, 43, 55]
ignore_point_ids = paper_specified_ignore + data_specified_ignore
calib_point_id = 61
start_year = 2006
end_year = 2006
norm_per_year = False
multi_threaded = False
ddt_scale = False
use_geo = False
sqrt_ddt_correction = False

df_temp = prepare_temp(temp_file, start_year, end_year, norm_per_year)
df_peak_alt = prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp)
df_alt_gt = df_peak_alt.groupby("point_id").mean()

def get_unique(df_alt_gt, col):
    vals = pd.unique(df_alt_gt[col])
    assert len(vals) == 1
    v = vals[0]
    return int(v)

year = get_unique(df_alt_gt, 'year')
month = get_unique(df_alt_gt, 'month')
day = get_unique(df_alt_gt, 'day')

si = igrams_usable
n = len(si)
lhs_all = np.zeros((n, len(df_alt_gt)))
rhs_all = np.zeros((n, 2))

all_scenes = []
for (scene1, scene2) in si:
    all_scenes.append(scene1)
    all_scenes.append(scene2)
all_scenes = set(all_scenes)

target_date = datetime(year, month, day)
all_scenes = [(get_date_for_alos(s)[1], s) for s in all_scenes]
all_scenes = sorted(all_scenes, key=lambda x: abs((x[0] - target_date)).days)
(calib_date, calib_scene) = all_scenes[0]
print("Calibration date, scene:", calib_date, calib_scene)

avg_alt_calib = df_alt_gt['alt_m'].mean()
avg_sqrt_norm_ddt_calib = np.sqrt(df_alt_gt['norm_ddt'].values).mean()

subs = df_alt_gt['alt_m'].apply(alt_to_surface_deformation)
avg_subsidence_stddev = np.std(subs)

def get_scene_expected_alt(scene):
    scene_d = get_date_for_alos(scene)[1]
    scene_sqrt_ddt = np.sqrt(df_temp.loc[scene_d.year, scene_d.month, scene_d.day]['norm_ddt'])
    avg_scene_alt = avg_alt_calib * scene_sqrt_ddt / avg_sqrt_norm_ddt_calib
    return avg_scene_alt

def get_expected_alt_per_pixel(scene):
    scene_d = get_date_for_alos(scene)[1]
    scene_sqrt_ddt = np.sqrt(df_temp.loc[scene_d.year, scene_d.month, scene_d.day]['norm_ddt'])
    expected_pixel_alt = df_alt_gt['alt_m'].values * scene_sqrt_ddt / np.sqrt(df_alt_gt['norm_ddt'].values)
    return expected_pixel_alt

# expected_N_per_pixel = np.square(df_alt_gt['alt_m'].values /np.sqrt(df_alt_gt['norm_ddt'].values))/2
lhs_all = np.zeros((n, len(df_alt_gt)))
rhs_all = np.zeros((n, 2))
for i, (scene1, scene2) in enumerate(si):
    scene1_avg_alt = get_scene_expected_alt(scene1)
    scene2_avg_alt = get_scene_expected_alt(scene2)
    scene1_avg_def = alt_to_surface_deformation(scene1_avg_alt)
    scene2_avg_def = alt_to_surface_deformation(scene2_avg_alt)
    expected_avg_def = scene1_avg_def - scene2_avg_def

    # IDEALIZED MODEL:
    scene1_expected_alt_per_pixel = get_expected_alt_per_pixel(scene1)
    scene1_expected_deformation_per_pixel = np.array([alt_to_surface_deformation(alt) for alt in scene1_expected_alt_per_pixel])
    scene2_expected_alt_per_pixel = get_expected_alt_per_pixel(scene2)
    scene2_expected_deformation_per_pixel = np.array([alt_to_surface_deformation(alt) for alt in scene2_expected_alt_per_pixel])
    deformation_per_pixel = scene1_expected_deformation_per_pixel - scene2_expected_deformation_per_pixel
    _deformation_per_pixel, rhs = process_scene_pair(
        scene1,
        scene2,
        df_alt_gt,
        None,
        df_temp,
        use_geo,
        sqrt_ddt_correction,
        False,
        # ideal_deformation=expected_deformation_per_pixel
    )
    deformation_per_pixel = deformation_per_pixel - deformation_per_pixel.mean() + expected_avg_def
    # lhs = np.sqrt(2*expected_N_per_pixel) * rhs[1]
    scene1_avg_def = alt_to_surface_deformation(scene1_avg_alt)
    scene2_est_deformation_per_pixel = scene1_avg_def - deformation_per_pixel
    scene2_est_alt_per_pixel = np.array([compute_alt_f_deformation(sub) for sub in scene2_est_deformation_per_pixel])
    lhs = scene1_avg_alt - scene2_est_alt_per_pixel
    
    lhs_all[i] = lhs
    rhs_all[i, :] = rhs

# Ignore time column, just DDT
rhs_all = rhs_all[:, [1]]
rhs_pi = np.linalg.pinv(rhs_all)
sol = rhs_pi @ lhs_all

alt_pred = sol[0, :]

# TODO: we should really do sqrt, avg per year, then overall avg
avg_sqrt_ddt = np.sqrt(df_alt_gt['norm_ddt'].values).mean()
alt_pred = alt_pred * avg_sqrt_ddt

print("AVG ALT", np.mean(alt_pred))

alt_gt = df_alt_gt["alt_m"].values
compute_stats(alt_pred, alt_gt)

print()
