
import datetime
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from pp.methods.resalt import ReSALT, ReSALT_Type, find_best_thaw_depth_difference
from pp.methods.soil_models import ConstantWaterSMM, LiuSMM, SoilMoistureModel


@pytest.mark.parametrize("Q", [1/2, 3])
def test_find_best_thaw_depth_difference(Q):
    """
    Tests that thaw depth difference retrieval (Algorithm 1) works.
    """
    smm = LiuSMM()
    
    alt1_large = 0.2
    alt2_large = alt1_large * Q
    sub1_large = smm.deformation_from_alt(alt1_large)
    sub2_large = smm.deformation_from_alt(alt2_large)
    sub_large = sub2_large - sub1_large
    expected_thaw_depth_diff_large = alt2_large - alt1_large
    
    alt1_small = 0.01
    alt2_small = alt1_small * Q
    sub1_small = smm.deformation_from_alt(alt1_small)
    sub2_small = smm.deformation_from_alt(alt2_small)
    sub_small = sub2_small - sub1_small
    expected_thaw_depth_diff_small = alt2_small - alt1_small
    
    best_thaw_depth_differences = find_best_thaw_depth_difference([sub_large, sub_small], Q, 1, smm)
    assert abs(best_thaw_depth_differences[0] - expected_thaw_depth_diff_large) < 1e-3
    assert abs(best_thaw_depth_differences[1] - expected_thaw_depth_diff_small) < 1e-3


def make_simulated_data(n_igrams: int, n_pixels: int, smm: SoilMoistureModel):
    rand = np.random.RandomState(7)
    
    # TODO: add R to simulation
    R = 0.0 #rand.uniform(0.001, 0.003)
    years = [2006] #[2006, 2007, 2008, 2009, 2010]
    ddt_per_year = []
    for _ in years:
        ddt_growth_per_day = rand.uniform(0.25, 0.4)
        # not accounting for leap years but nothing really cares about that
        ddt_per_day = np.arange(365) * ddt_growth_per_day
        ddt_per_year.append(ddt_per_day)
        
    def construct_datetime(year, day_of_year):
        return datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    
    deformations = []
    dates = []
    # Chosen to get realistic ALTs (like in Barrow). This is k_p/(p_p * L) in the Appendix.
    soil_thaw_coeff_mean = 7.5e-4
    soil_thaw_coeff_std = 2.5e-4
    soil_thaw_coeffs = np.clip(rand.randn(n_pixels), -1.5, 1.5) * soil_thaw_coeff_std + soil_thaw_coeff_mean
    while len(deformations) != n_igrams:
        year_ref_idx = rand.randint(len(years))
        year_ref = years[year_ref_idx]
        day_ref = rand.randint(1, 366)
        year_sec_idx = rand.randint(len(years))
        year_sec = years[year_sec_idx]
        day_sec = rand.randint(1, 366)
        
        deformation_row = []
        for i in range(n_pixels):
            ddt_ref = ddt_per_year[year_ref_idx][day_ref - 1]
            seasonal_alt_ref = np.sqrt(2*soil_thaw_coeffs[i]*ddt_ref)
            seasonal_def_ref = smm.deformation_from_alt(seasonal_alt_ref)
            
            ddt_sec = ddt_per_year[year_sec_idx][day_sec - 1]
            seasonal_alt_sec = np.sqrt(2*soil_thaw_coeffs[i]*ddt_sec)
            seasonal_def_sec = smm.deformation_from_alt(seasonal_alt_sec)
            
            net_deformation = R*(year_ref - year_sec + (day_ref - day_sec)/365) + seasonal_def_ref - seasonal_def_sec
            deformation_row.append(net_deformation)
        
        deformations.append(deformation_row)
        dates.append((construct_datetime(year_ref, day_ref), construct_datetime(year_sec, day_sec)))
        
    deformations = np.array(deformations)
    df_temps = []
    alts_per_pixel = []
    for i in range(len(years)):
        df = pd.DataFrame()
        df['ddt'] = ddt_per_year[i]
        months = []
        days = []
        for doy in range(1, len(ddt_per_year[i]) + 1):
            dt = construct_datetime(years[i], doy)
            months.append(dt.month)
            days.append(dt.day)
        df['month'] = months
        df['day'] = days
        df['year'] = years[i]
        df_temps.append(df)
        alt_row = []
        for j in range(len(soil_thaw_coeffs)):
            max_year_alt = np.sqrt(2*soil_thaw_coeffs[j]*ddt_per_year[i][-1])
            alt_row.append(max_year_alt)
        alts_per_pixel.append(alt_row)
    df_temp = pd.concat(df_temps)
    max_ddt = df_temp["ddt"].max()
    df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt
    df_temp = df_temp.set_index(['year', 'month', 'day'])
    avg_cross_year_alt_per_pixel = np.mean(alts_per_pixel, axis=0)
    print("Avg ALT:", np.mean(avg_cross_year_alt_per_pixel), "STD ALT:", np.std(avg_cross_year_alt_per_pixel))
    return deformations, dates, df_temp, avg_cross_year_alt_per_pixel    


def test_constant_smm():
    """
    Tests a soil model which should perform perfectly for ReSALT and SCReSALT.
    """
    smm = ConstantWaterSMM(0.5)
    n_igrams = 10
    n_pixels = 20
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm)
    calib_idx = 0
    calib_deformation = smm.deformation_from_alt(alt_gt[calib_idx])
    # Ignore calibration point
    alt_gt = alt_gt[1:]
    
    # 1. Test normal ReSALT
    rtype = ReSALT_Type.LIU_SCHAEFER
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    
    assert rmse < 1e-3
    
    # 2. Test SCReSALT
    rtype = ReSALT_Type.SCReSALT
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    
    assert rmse < 1e-3
    

def test_liu_smm():
    """
    Tests a soil model which ReSALT should fail for but SCReSALT properly handles.
    """
    smm = LiuSMM()
    n_igrams = 20
    n_pixels = 100
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm)
    calib_idx = 0
    calib_deformation = smm.deformation_from_alt(alt_gt[calib_idx])
    # Ignore calibration point
    alt_gt = alt_gt[1:]
    
    # 1. Test normal ReSALT
    rtype = ReSALT_Type.LIU_SCHAEFER
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    
    # Non-trivial RMSE for normal ReSALT because it inconsistently handles LiuSMM (as discussed in paper)
    assert rmse > 1e-2
    
    # 2. Test SCReSALT
    rtype = ReSALT_Type.SCReSALT
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    
    # Much lower RMSE for SCReSALT because it consistently handles LiuSMM
    assert rmse < 1e-3
    
