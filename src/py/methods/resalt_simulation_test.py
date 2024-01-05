

import datetime
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


sys.path.append("/permafrost-prediction/src/py")
from methods.resalt import ReSALT, ReSALT_Type
from methods.soil_models import ConstantWaterSMM, LiuSMM, SoilMoistureModel


# TODO: DDT errors account for a lot of error. Handle it?
def make_simulated_data(n_igrams: int, n_pixels: int, smm: SoilMoistureModel):
    rand = np.random.RandomState(7)
    
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
    N_mean = 7.5e-4
    N_std = 2.5e-4
    Ns = np.clip(rand.randn(n_pixels), -1.5, 1.5) * N_std + N_mean
    while len(deformations) != n_igrams:
        year_ref_idx = rand.randint(len(years))
        year_ref = years[year_ref_idx]
        day_ref = rand.randint(1, 366)
        year_sec_idx = rand.randint(len(years))
        year_sec = years[year_sec_idx]
        day_sec = rand.randint(1, 366)
        
        day_ref = 365
        deformation_row = []
        for i in range(n_pixels):
            ddt_ref = ddt_per_year[year_ref_idx][day_ref - 1]
            seasonal_alt_ref = np.sqrt(2*Ns[i]*ddt_ref)
            seasonal_def_ref = smm.deformation_from_alt(seasonal_alt_ref)
            
            ddt_sec = ddt_per_year[year_sec_idx][day_sec - 1]
            seasonal_alt_sec = np.sqrt(2*Ns[i]*ddt_sec)
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
        for j in range(len(Ns)):
            max_year_alt = np.sqrt(2*Ns[j]*ddt_per_year[i][-1])
            alt_row.append(max_year_alt)
        alts_per_pixel.append(alt_row)
    df_temp = pd.concat(df_temps)
    max_ddt = df_temp["ddt"].max()
    df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt
    df_temp = df_temp.set_index(['year', 'month', 'day'])
    avg_cross_year_alt_per_pixel = np.mean(alts_per_pixel, axis=0)
    print("Avg ALT:", np.mean(avg_cross_year_alt_per_pixel), "STD ALT:", np.std(avg_cross_year_alt_per_pixel))
    return deformations, dates, df_temp, avg_cross_year_alt_per_pixel


def test_constant_smm(rtype: ReSALT_Type):
    smm = ConstantWaterSMM(s=0.5)
    n_igrams = 10
    n_pixels = 20
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm)
    calib_idx = 0
    calib_deformation = smm.deformation_from_alt(alt_gt[calib_idx])
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    
    # Ignore calibration point for errors
    alt_pred = alt_pred[1:]
    alt_gt = alt_gt[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    # This bound could be tighter, but we do discrete sampling to solve Jatin variety which induces sampling errors.
    assert rmse < 5e-3
    # pearson_corr, _ = pearsonr(alt_pred, alt_gt)
    
    # print("RMSE", rmse)
    # print("Pearson R", pearson_corr)
    
    
def test_liu_smm(rtype: ReSALT_Type, plot=False):
    smm = LiuSMM()
    n_igrams = 20
    n_pixels = 100
    deformations, dates, df_temp, alt_gt = make_simulated_data(n_igrams, n_pixels, smm)
    calib_idx = 0
    calib_deformation = smm.deformation_from_alt(alt_gt[calib_idx])
    resalt = ReSALT(df_temp, smm, calib_idx, calib_deformation, rtype)
    alt_pred = resalt.run_inversion(deformations, dates)
    
    # Ignore calibration point for errors
    alt_pred = alt_pred[1:]
    alt_gt = alt_gt[1:]
    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    print(f"Resalt {rtype} RMSE under Liu model:", round(rmse, 4))
    
    
    
if __name__ == '__main__':
    # test_constant_smm(ReSALT_Type.JATIN)
    test_liu_smm(ReSALT_Type.JATIN)
    
