
from abc import ABC, abstractmethod
import datetime
from typing import List, Type
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from pp.methods.resalt import ReSALT, ReSALT_Type, scresalt_find_best_thaw_depth_difference
from pp.methods.soil_models import ConstantWaterSMM, LiuSMM, SoilDepthIntegration, SoilMoistureModel


@pytest.mark.parametrize("sqrt_ddt_ratio", [1/2, 3])
def test_find_best_thaw_depth_difference(sqrt_ddt_ratio):
    """
    Tests that thaw depth difference retrieval (Algorithm 1) works.
    """
    smm = LiuSMM()
    
    alt1_large = 0.2
    alt2_large = alt1_large * sqrt_ddt_ratio
    sub1_large = smm.deformation_from_thaw_depth(alt1_large)
    sub2_large = smm.deformation_from_thaw_depth(alt2_large)
    sub_large = sub2_large - sub1_large
    expected_thaw_depth_diff_large = alt2_large - alt1_large
    
    alt1_small = 0.01
    alt2_small = alt1_small * sqrt_ddt_ratio
    sub1_small = smm.deformation_from_thaw_depth(alt1_small)
    sub2_small = smm.deformation_from_thaw_depth(alt2_small)
    sub_small = sub2_small - sub1_small
    expected_thaw_depth_diff_small = alt2_small - alt1_small
    
    best_thaw_depth_differences = scresalt_find_best_thaw_depth_difference([sub_large, sub_small], sqrt_ddt_ratio, smm, 1.0, 1000)
    assert abs(best_thaw_depth_differences[0] - expected_thaw_depth_diff_large) < 1e-3
    assert abs(best_thaw_depth_differences[1] - expected_thaw_depth_diff_small) < 1e-3


def make_simulated_data(n_igrams: int, n_pixels: int, smm: SoilMoistureModel, cls: Type["SoilDynamics"]):
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
    ddt_per_year = np.array(ddt_per_year)
    ddt_per_year = ddt_per_year / np.max(ddt_per_year) # normalize
        
    def construct_datetime(year, day_of_year):
        return datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)
    
    deformations = []
    dates = []
    
    ssds: List[SoilDynamics] = []
    for _ in range(n_pixels):
        ssds.append(cls(smm, np.clip(rand.randn(1)[0], -1.5, 1.5)))
    
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
            seasonal_alt_ref = ssds[i].generate_thaw_depth_from_ddt(ddt_ref)
            seasonal_def_ref = smm.deformation_from_thaw_depth(seasonal_alt_ref)
            
            ddt_sec = ddt_per_year[year_sec_idx][day_sec - 1]
            seasonal_alt_sec = ssds[i].generate_thaw_depth_from_ddt(ddt_sec)
            seasonal_def_sec = smm.deformation_from_thaw_depth(seasonal_alt_sec)
            
            net_deformation = R*(year_ref - year_sec + (day_ref - day_sec)/365) + seasonal_def_ref - seasonal_def_sec
            deformation_row.append(net_deformation)
        
        deformations.append(deformation_row)
        dates.append((construct_datetime(year_ref, day_ref), construct_datetime(year_sec, day_sec)))
        
    deformations = np.array(deformations)
    df_temps = []
    alts_per_pixel = []
    for i in range(len(years)):
        df = pd.DataFrame()
        df['norm_ddt'] = ddt_per_year[i]
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
        for ssd in ssds:
            # Use end-of-year ALT
            max_year_alt = ssd.generate_thaw_depth_from_ddt(ddt_per_year[i][-1])
            alt_row.append(max_year_alt)
        alts_per_pixel.append(alt_row)
    df_temp = pd.concat(df_temps)
    df_temp = df_temp.set_index(['year', 'month', 'day'])
    avg_cross_year_alt_per_pixel = np.mean(alts_per_pixel, axis=0)
    print("Avg ALT:", np.mean(avg_cross_year_alt_per_pixel), "STD ALT:", np.std(avg_cross_year_alt_per_pixel))
    return deformations, dates, df_temp, avg_cross_year_alt_per_pixel


class SoilDynamics(ABC):
    @abstractmethod
    def __init__(self, smm: SoilMoistureModel, sigma):
        pass
    
    @abstractmethod
    def generate_thaw_depth_from_ddt(self, ddt):
        pass

class StefanSoilDynamics(SoilDynamics):
    def __init__(self, smm: SoilMoistureModel, sigma):
        # Chosen to get realistic ALTs (like in Barrow). This is N = k_p/(p_p * L) in the Appendix (for normalized DDTs).
        soil_thaw_coeff_mean = 7.5e-2
        soil_thaw_coeff_std = 2.5e-2
        self.N = 2 * (sigma * soil_thaw_coeff_std + soil_thaw_coeff_mean)
    
    def generate_thaw_depth_from_ddt(self, ddt):
        return np.sqrt(self.N*ddt)
    
    
class NonStefanSoilDynamics(SoilDynamics):
    """Captures effect of non-constant porosity model. Uses depth-porosity integrated function to relate thaw depth and temperature."""
    def __init__(self, smm: SoilMoistureModel, sigma):
        # Chosen to get realistic ALTs. This is M in the Appendix (for normalized DDTs).
        M_mean = 0.05
        M_std = 0.01
        self.M = sigma * M_std + M_mean
        self.smm = smm
        # This helps reduce computational runtime by calculating the integral ahead of time.
        self.sdi = SoilDepthIntegration(smm, 1.0, 1000)
        
    def generate_thaw_depth_from_ddt(self, ddt):
        td = self.sdi.find_thaw_depth_for_integral(ddt * self.M)
        assert td is not None, "Thaw depth max was exceeded"
        return td.h
    

@pytest.mark.parametrize("ssd", [StefanSoilDynamics, NonStefanSoilDynamics])
def test_constant_smm(ssd):
    """
    The constant porosity model should perform perfectly for ReSALT, SCReSALT, and SCReSALT-NS.
    This should be true regardless of the soil dynamics model used, as NonStefanSoilDynamics
    simplifies to StefanSoilDynamics.
    """
    smm = ConstantWaterSMM(0.5)
    n_igrams = 10
    n_pixels = 20
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm, ssd)
    calib_idx = 0
    calib_deformation = smm.deformation_from_thaw_depth(alt_gt[calib_idx])
    # ALT is for end of year, so get end of year DDTs. TODO: encode this better?
    calib_ddts = df_temp.groupby('year').last()['norm_ddt'].values
    # Ignore calibration point
    alt_gt = alt_gt[1:]
    
    # 1. Test normal ReSALT
    rtype = ReSALT_Type.LIU_SCHAEFER
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    assert rmse < 1e-3
    
    # 2. Test SCReSALT
    rtype = ReSALT_Type.SCReSALT
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    assert rmse < 1e-3
    
    # 3. Test SCReSALT-NS
    rtype = ReSALT_Type.SCReSALT_NS
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    
    assert rmse < 1e-3
    

def test_liu_smm_with_stefan_soil_dynamics():
    """
    Tests a soil model which ReSALT should fail for but SCReSALT properly handles.
    """
    smm = LiuSMM()
    n_igrams = 20
    n_pixels = 100
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm, StefanSoilDynamics)
    calib_idx = 0
    calib_deformation = smm.deformation_from_thaw_depth(alt_gt[calib_idx])
    # ALT is for end of year, so get end of year DDTs. TODO: encode this better?
    calib_ddts = df_temp.groupby('year').last()['norm_ddt'].values
    
    # Ignore calibration point
    alt_gt = alt_gt[1:]
    
    # 1. Test normal ReSALT
    rtype = ReSALT_Type.LIU_SCHAEFER
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    # Non-trivial RMSE for normal ReSALT because it inconsistently handles LiuSMM (as shown in paper)
    assert rmse > 1e-2
    
    # 2. Test SCReSALT
    rtype = ReSALT_Type.SCReSALT
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    # Much lower RMSE for SCReSALT because it consistently handles LiuSMM
    assert rmse < 1e-3
    

def test_liu_smm_with_nonstefan_soil_dynamics():
    smm = LiuSMM()
    n_igrams = 20
    n_pixels = 100
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm, NonStefanSoilDynamics)
    calib_idx = 0
    calib_deformation = smm.deformation_from_thaw_depth(alt_gt[calib_idx])
    # ALT is for end of year, so get end of year DDTs. TODO: encode this better?
    calib_ddts = df_temp.groupby('year').last()['norm_ddt'].values
    
    # Ignore calibration point
    alt_gt = alt_gt[1:]
    
    # 1. Test normal ReSALT
    rtype = ReSALT_Type.LIU_SCHAEFER
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse_resalt = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    # Non-trivial RMSE for normal ReSALT because it inconsistently handles LiuSMM (as shown in paper)
    assert rmse_resalt > 5e-3
    
    # 2. Test SCReSALT
    rtype = ReSALT_Type.SCReSALT
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse_scresalt = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    # Lower RMSE for SCReSALT, but should not be perfect because soil dynamics are non-Stefan.
    assert 3e-3 < rmse_scresalt < rmse_resalt
    
    # 3. Test SCReSALT-NS
    rtype = ReSALT_Type.SCReSALT_NS
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, calib_ddts, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    # Ignore calibration point
    alt_pred = alt_pred[1:]
    rmse_scresalt_ns = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    # Perfectly handles data.
    assert rmse_scresalt_ns < 1e-3
    assert rmse_scresalt_ns < rmse_scresalt

