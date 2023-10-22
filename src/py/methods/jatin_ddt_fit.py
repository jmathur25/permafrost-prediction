
from datetime import date, datetime
import sys

import numpy as np
import pandas as pd


sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import alt_to_surface_deformation
from methods.schaefer import process_scene_pair
from data.utils import get_date_for_alos
from methods.utils import prepare_calm_data, prepare_temp
from methods.igrams import JATIN_SINGLE_SEASON_2006_IGRAMS
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR

igrams_usable = []
for (alos2, alos1) in JATIN_SINGLE_SEASON_2006_IGRAMS:
    d = ISCE2_OUTPUTS_DIR / f"{alos2}_{alos1}"
    if not (d / 'interferogram/filt_topophase.unw').exists():
        print("TODO:", d)
    else:
        igrams_usable.append((alos2, alos1))
        
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

def get_scene_patch_mean_sub(scene):
    scene_d = get_date_for_alos(scene)[1]
    scene_sqrt_ddt = np.sqrt(df_temp.loc[scene_d.year, scene_d.month, scene_d.day]['norm_ddt'])
    avg_scene_alt = avg_alt_calib * scene_sqrt_ddt / avg_sqrt_norm_ddt_calib
    return alt_to_surface_deformation(avg_scene_alt)

lhs_all = np.zeros((n, len(df_alt_gt)))
rhs_all = np.zeros((n, 2))
for i, (scene1, scene2) in enumerate(si):
    scene1_patch_mean = get_scene_patch_mean_sub(scene1)
    scene2_patch_mean = get_scene_patch_mean_sub(scene2)
    lhs, rhs = process_scene_pair(
        scene1,
        scene2,
        df_alt_gt,
        None,
        df_temp,
        use_geo,
        sqrt_ddt_correction,
        False
    )
    # Make the average the expected subsidence difference
    # TODO: should this be done on the entire slice?
    lhs = np.array(lhs)
    lhs = lhs - lhs.mean() + (scene1_patch_mean - scene2_patch_mean)
    lhs_all[i] = lhs
    rhs_all[i, :] = rhs

# Ignore time column, just DDT
rhs_all = rhs_all[i, 1]
rhs_pi = np.linalg.pinv(rhs_all)
sol = rhs_pi @ lhs_all

E = sol[0, :]

print()
