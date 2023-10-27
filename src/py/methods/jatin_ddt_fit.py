
from datetime import date, datetime
import sys

import numpy as np
import pandas as pd




sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import LiuSMM
from methods.simulate_sub_diff_solve import find_best_alt_diff
from scaling_theory import RefBiasDirection, estimate_alts
from methods.schaefer import process_scene_pair
from data.utils import get_date_for_alos
from methods.utils import compute_stats, prepare_calm_data, prepare_temp
from methods.igrams import JATIN_SINGLE_SEASON_2006_IGRAMS, SCHAEFER_INTEFEROGRAMS
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR


def get_expected_calib_alt(scene_sqrt_ddt, df_alt_gt, calib_point_id, year: int):
    row = df_alt_gt.loc[calib_point_id, year]
    calib_sqrt_ddt = np.sqrt(row['norm_ddt'])
    calib_alt = row['alt_m']
    avg_scene_alt = calib_alt * scene_sqrt_ddt / calib_sqrt_ddt
    return avg_scene_alt


def get_scene_ddt(scene, df_temp):
    scene_d = get_date_for_alos(scene)[1]
    scene_sqrt_ddt = np.sqrt(df_temp.loc[scene_d.year, scene_d.month, scene_d.day]['norm_ddt'])
    return scene_sqrt_ddt


def get_expected_alt_per_pixel(scene, df_temp, df_alt_gt):
    scene_d = get_date_for_alos(scene)[1]
    scene_sqrt_ddt = np.sqrt(df_temp.loc[scene_d.year, scene_d.month, scene_d.day]['norm_ddt'])
    expected_pixel_alt = df_alt_gt['alt_m'].values * scene_sqrt_ddt / np.sqrt(df_alt_gt['norm_ddt'].values)
    return expected_pixel_alt

def solve_jatin_resalt_reformulated():
    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

    paper_specified_ignore = [7, 110, 121]
    data_specified_ignore = [21, 43, 55] + [8, 9, 20, 34, 45, 56, 67, 68, 78, 89, 90, 100, 101, 102, 111]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    start_year = 2006
    end_year = 2009
    calib_point_id = 61
    use_calib_node = True
    multi_threaded = False
    ddt_scale = False
    use_geo = False
    sqrt_ddt_correction = False

    df_temp = prepare_temp(temp_file, start_year, end_year)
    df_alt_gt = prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp)
    df_calm_points = df_alt_gt[['point_id', 'latitude', 'longitude']].groupby('point_id').first()
    df_alt_gt = df_alt_gt.set_index(['point_id', 'year'])
    # df_alt_gt = df_alt_gt.groupby("point_id").mean()

    # def get_unique(df_alt_gt, col):
    #     vals = pd.unique(df_alt_gt[col])
    #     assert len(vals) == 1
    #     v = vals[0]
    #     return int(v)

    # year = get_unique(df_alt_gt, 'year')
    # month = get_unique(df_alt_gt, 'month')
    # day = get_unique(df_alt_gt, 'day')
    # print("RENORMALIZING TO MEASUREMENT")
    # measurement_ddt = df_temp.loc[year, month, day]['norm_ddt']
    # df_temp['norm_ddt'] = df_temp['norm_ddt'] / measurement_ddt
    # df_temp[df_temp['norm_ddt'] > 1.0] = 1.0
    # df_alt_gt['norm_ddt'] = df_alt_gt['norm_ddt'] / measurement_ddt
    # df_alt_gt[df_alt_gt['norm_ddt'] > 1.0] = 1.0

    si = [
        ("ALPSRP021272170", "ALPSRP027982170"),
        ("ALPSRP074952170", "ALPSRP081662170"),
        ("ALPSRP182312170", "ALPSRP189022170"),
        # ("ALPSRP235992170", "ALPSRP242702170") TODO: run
    ]
    needs_sign_flip = True
    n = len(si)
    smm = LiuSMM()

    # all_scenes = []
    # for (scene1, scene2) in si:
    #     all_scenes.append(scene1)
    #     all_scenes.append(scene2)
    # all_scenes = set(all_scenes)
    # TODO: work-in average approach
    # target_date = datetime(year, month, day)
    # all_scenes = [(get_date_for_alos(s)[1], s) for s in all_scenes]
    # all_scenes = sorted(all_scenes, key=lambda x: abs((x[0] - target_date)).days)
    # (calib_date, calib_scene) = all_scenes[0]
    # print("Calibration date, scene:", calib_date, calib_scene)

    # avg_alt = df_alt_gt['alt_m'].mean()
    # avg_sqrt_norm_ddt = np.sqrt(df_alt_gt['norm_ddt'].values).mean()

    # if use_calib_node:
    #     print("Calibrating using calibration node:", calib_point_id)
    # else:
    #     print("Calibrating using average")
    
    lhs_all = np.zeros((n, len(df_calm_points)))
    rhs_all = np.zeros((n, 2))
    for i, (scene1, scene2) in enumerate(si):
        if needs_sign_flip:
            scene2, scene1 = scene1, scene2
            
        scene1_date = get_date_for_alos(scene1)[1]
        scene2_date = get_date_for_alos(scene2)[1]
        # Only formulated where scene1 and scene2 are in the same year, and scene1 is after scene2
        assert scene1_date.year == scene2_date.year
        assert scene1_date.month > scene2_date.month
        scene1_sqrt_ddt = get_scene_ddt(scene1, df_temp)
        scene2_sqrt_ddt = get_scene_ddt(scene2, df_temp)
        if use_calib_node:
            scene1_calib_alt = get_expected_calib_alt(scene1_sqrt_ddt, df_alt_gt, calib_point_id, scene1_date.year)
            scene2_calib_alt = get_expected_calib_alt(scene2_sqrt_ddt, df_alt_gt, calib_point_id, scene2_date.year)
            # GETS PARITY:
            # scene1_calib_alt=0.3076322679068533*scene1_sqrt_ddt
            # scene2_calib_alt=0.3076322679068533*scene2_sqrt_ddt
        else:
            # TODO: implement averaging approach
            raise ValueError()
            # scene1_calib_alt = avg_alt * scene1_sqrt_ddt / avg_sqrt_norm_ddt
            # scene2_calib_alt = avg_alt * scene2_sqrt_ddt / avg_sqrt_norm_ddt
        scene1_calib_def = smm.deformation_from_alt(scene1_calib_alt)
        scene2_calib_def = smm.deformation_from_alt(scene2_calib_alt)
        expected_calib_def = scene1_calib_def - scene2_calib_def

        # Flip just for invoking process_scene_pair
        if needs_sign_flip:
            scene2, scene1 = scene1, scene2
        deformation_per_pixel, rhs = process_scene_pair(
            scene1,
            scene2,
            df_calm_points,
            None,
            df_temp,
            use_geo,
            sqrt_ddt_correction,
            needs_sign_flip=needs_sign_flip,
            # ideal_deformation=expected_deformation_per_pixel
        )
        if needs_sign_flip:
            scene2, scene1 = scene1, scene2
            
        if use_calib_node:
            matches = np.argwhere(df_calm_points.index==calib_point_id)
            assert matches.shape == (1,1)
            # The delta needed to make the calibration node have the right deformation
            calib_node_delta = expected_calib_def - deformation_per_pixel[matches[0,0]]
            deformation_per_pixel = deformation_per_pixel + calib_node_delta
        else:
            deformation_per_pixel = deformation_per_pixel - deformation_per_pixel.mean() + expected_calib_def
        
        # scene2_est_alt_per_pixel = np.array([compute_alt_f_deformation(sub) if sub > 1e-3 else np.nan for sub in scene2_est_deformation_per_pixel])
        # scene1_est_alt_per_pixel = scene1_avg_alt/scene2_avg_alt * scene2_est_alt_per_pixel
        # lhs = scene1_est_alt_per_pixel - scene2_est_alt_per_pixel
        
        # TODO: OLD
        # alt_pred_later, alt_pred_earlier = estimate_alts(deformation_per_pixel, scene1_calib_alt, scene2_sqrt_ddt/scene1_sqrt_ddt, RefBiasDirection.NONE)
        # lhs = alt_pred_later - alt_pred_earlier
        
        lhs = find_best_alt_diff(deformation_per_pixel, scene1_sqrt_ddt, scene2_sqrt_ddt, smm)
        
        lhs_all[i] = lhs
        rhs_all[i, :] = rhs

    # Ignore time column, just DDT
    rhs_all = rhs_all[:, [1]]
    rhs_pi = np.linalg.pinv(rhs_all)
    sol = rhs_pi @ lhs_all
    alt_pred = sol[0, :]

    nans = np.argwhere(np.isnan(alt_pred))
    print("Number of pixels with nans in least-squares inversion:", len(nans))
    for i in nans[:,0]:
        pixel_id = df_calm_points.index[i]
        lhs_i = lhs_all[:,i]
        nan_mask = np.isnan(lhs_i)
        if np.mean(nan_mask) > 0.5:
            print(f"Pixel {pixel_id} has too many nans. Skipping inversion.")
            continue
        not_nan_mask = ~nan_mask
        alt_i = np.linalg.pinv(rhs_all[not_nan_mask]) @ lhs_i[not_nan_mask]
        alt_pred[i] = alt_i
        print(f"Resolved pixel {pixel_id} with a subset of non-nan data")

    df_alt_gt['sqrt_norm_ddt'] = np.sqrt(df_alt_gt['norm_ddt'].values)
    df_alt_avg = df_alt_gt[['alt_m', 'sqrt_norm_ddt']].groupby(['point_id']).mean()
    alt_pred = alt_pred * df_alt_avg['sqrt_norm_ddt'].values
    alt_gt = df_alt_avg['alt_m'].values

    compute_stats(alt_pred, alt_gt)

    print()
    

if __name__ == '__main__':
    solve_jatin_resalt_reformulated()
