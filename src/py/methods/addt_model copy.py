from datetime import datetime
import sys
import click
import numpy as np
import pandas as pd

sys.path.append("/permafrost-prediction/src/py")
from methods.utils import compute_ddt_ddf, prepare_calm_data, prepare_temp
from data.consts import CALM_PROCESSSED_DATA_DIR, TEMP_DATA_DIR

from scipy.stats import pearsonr

@click.command()
def addt_method():
    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    def extract_month_day(row):
        date = datetime.strptime(f"{row['YEAR']}-{row['DAY']}", '%Y-%j')
        return date.month, date.day
    dfs = []
    norm_per_year = True
    for i in range(6, 11):
        istr = str(i).zfill(2)
        df = pd.read_excel(f"/permafrost-prediction-shared-data/Barrow1_{istr}ave.xls")
        assert df['MRC.2'].values[0] == '5 cm'
        df = df.dropna(subset='YEAR')
        df['YEAR'] = df['YEAR'].astype(int)
        df['DAY'] = df['DAY'].astype(int)
        df['month'], df['day'] = zip(*df.apply(extract_month_day, axis=1))
        df = df.rename({'YEAR': 'year', 'MRC.2': 'temp_2m_c'}, axis=1)
        compute_ddt_ddf(df)
        cols = ['year', 'month', 'day', 'temp_2m_c', 'ddt']
        if norm_per_year:
            df['norm_ddt'] = df['ddt'] / df['ddt'].values[-1]
            cols.append('norm_ddt')
        dfs.append(df[cols])
    df_temp = pd.concat(dfs)
    if not norm_per_year:
        max_ddt = df_temp["ddt"].max()
        df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt
    df_temp = df_temp.set_index(['year', 'month', 'day'])
    
    paper_specified_ignore = [7, 110, 121]
    data_specified_ignore = [21, 43, 55]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    start_year = 2006
    end_year = 2010

    ddt_scale = False
    df_calm = prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp)
    
    alt_ratios = []
    ddt_ratios = []
    for (_, df) in df_calm.groupby('point_id'):
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                alt1 = df['alt_m'].values[i]
                alt2 = df['alt_m'].values[j]
                if np.isnan(alt1) or np.isnan(alt2):
                    continue

                ddt1 = df['norm_ddt'].values[i]
                ddt2 = df['norm_ddt'].values[j]
                
                alt_ratios.append(alt2/alt1)
                ddt_ratios.append(np.sqrt(ddt2/ddt1))
                
    
    
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
    
    pearson_r, _ = pearsonr(alt_ratios, ddt_ratios)
    print("CORR COEF", pearson_r)
    
    

if __name__ == "__main__":
    addt_method()
