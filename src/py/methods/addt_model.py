"""
Not relevant to paper. This is a simple baseline that takes 20% of the data on ADDT
and ALT and fits a line to it. While SAR is more 'hands-off' in that is only requires
a way to calibrate subsidence in the image and otherwise does not require any in-situ
data to fit against, this baseline is a good comparison. It might be more useful
in the future with a multiyear analysis (as ADDT is currently fixed per year for any site).
"""

import sys
import click
import numpy as np
import pandas as pd

sys.path.append("/permafrost-prediction/src/py")
from methods.utils import compute_stats, prepare_calm_data, prepare_temp
from data.consts import CALM_PROCESSSED_DATA_DIR, TEMP_DATA_DIR

import matplotlib.pyplot as plt

from scipy.stats import pearsonr

import logging

logging.getLogger().setLevel(logging.INFO)

@click.command()
def addt_method():
    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

    paper_specified_ignore = [7, 110, 121]
    data_specified_ignore = [21, 43, 55]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    start_year = 1995
    end_year = 2013

    df_temp = prepare_temp(temp_file, start_year, end_year)
    df_peak_alt = prepare_calm_data(
        calm_file, ignore_point_ids, start_year, end_year, df_temp
    )
    
    def print_stats(x, y, x_desc):
        pearson_r, _ = pearsonr(x, y)
        print(f"average yearly {x_desc} vs ALT pearson R:", pearson_r)
        
        plt.scatter(x, y)
        plt.ylabel("avg yearly alt_m")
        plt.xlabel(f"avg yearly {x_desc}")
        plt.savefig(f"addt_corr_{x_desc}.png")
        plt.close()
        
    x = df_peak_alt['sqrt_norm_ddt'].values
    y = df_peak_alt['alt_m'].values
    
    is_nan = np.isnan(x) | np.isnan(y)
    print("NANS", np.mean(is_nan))
    not_nan = ~is_nan
    x = x[not_nan]
    y = y[not_nan]
    
    print_stats(x, y, "sqrt_norm_ddt")
    
    frac_fit = 0.2
    selected_indices = np.random.choice(x.size, size=int(frac_fit * x.size), replace=False)
    mask = np.zeros_like(x, dtype=bool)
    mask[selected_indices] = True
    
    x_train = x[mask]
    y_train = y[mask]
    
    x_val = x[~mask]
    y_val = y[~mask]
    
    # no bias
    x_inv = np.linalg.pinv(x_train[:,None])
    sol = x_inv @ y_train
    
    y_train_pred = x_train * sol
    compute_stats(y_train_pred, y_train)
    
    y_val_pred = x_val * sol
    compute_stats(y_val_pred, y_val)
    
    # alt_ratios = []
    # ddt_ratios = []
    # for (_, df) in df_calm.groupby('point_id'):
    #     for i in range(len(df)):
    #         for j in range(i + 1, len(df)):
    #             alt1 = df['alt_m'].values[i]
    #             alt2 = df['alt_m'].values[j]
    #             if np.isnan(alt1) or np.isnan(alt2):
    #                 continue

    #             ddt1 = df['ddt'].values[i]
    #             ddt2 = df['ddt'].values[j]
                
    #             alt_ratios.append(alt2/alt1)
    #             ddt_ratios.append(np.sqrt(ddt2/ddt1))
                
                # alt_ratios +=[alt1, alt2]
                # ddt_ratios +=[ddt1, ddt2]
                
    
    
    # df_calm = pd.read_csv(calm_file, parse_dates=["date"])
    # df_calm = df_calm.sort_values("date", ascending=True)
    # df_calm = df_calm[df_calm["point_id"].apply(lambda x: x not in ignore_point_ids)]
    # df_calm = df_calm[(df_calm["date"] >= pd.to_datetime(str(start_year))) & (df_calm["date"] <= pd.to_datetime(str(end_year)))]

    # def try_float(x):
    #     try:
    #         return float(x)
    #     except:
    #         return np.nan

    # df_calm["alt_m"] = df_calm["alt_m"].apply(try_float) / 100
    # # only grab ALTs from end of summer, which willl be the last
    # # measurement in a year
    # df_calm["year"] = df_calm["date"].dt.year
    # df_calm['month'] = df_calm['date'].dt.month
    # df_calm['day'] = df_calm['date'].dt.day
    # df_calm = pd.merge(df_calm, df_temp[['norm_ddt']], on=['year', 'month', 'day'], how='left')
    # df_calm['day'] = df_calm['date'].dt.day
    
    # x = []
    # y = []
    # for ((_, year), df) in df_calm.groupby(["point_id", "year"]):
    #     if year != 2006:
    #         print()
            
    #     if len(df) == 1:
    #         continue
    #     first_alt = df['alt_m'].values[0]
    #     last_alt = df['alt_m'].values[-1]
        
    #     first_ddt = df['norm_ddt'].values[0]
    #     last_ddt = df['norm_ddt'].values[-1]
        
    #     x.append(last_alt - first_alt)
    #     y.append(last_ddt - first_ddt)
    
    # pearson_r, _ = pearsonr(alt_ratios, ddt_ratios)
    # print("CORR COEF", pearson_r)
    
    

if __name__ == "__main__":
    addt_method()
