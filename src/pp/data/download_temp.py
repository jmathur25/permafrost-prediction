"""
Downloads temperature data.
"""

import click
import numpy as np
import pandas as pd

from pp.data.consts import TEMP_DATA_DIR

# Data is downloaded from this URL
URL = "https://gml.noaa.gov/aftp/data/meteorology/in-situ/brw/met_brw_insitu_1_obop_hour_{}.txt"


@click.command()
@click.argument("start_year", type=int)
@click.argument("end_year", type=int)
def barrow_temperature(start_year, end_year):
    _download_barrow_temp_for_year(start_year, end_year)
    
def _download_barrow_temp_for_year(start_year, end_year):
    savepath = TEMP_DATA_DIR / "barrow/data/data.csv"
    years_to_download = set(range(start_year, end_year + 1))
    if savepath.exists():
        cur_df = pd.read_csv(savepath)
        downloaded_years = pd.unique(cur_df['year'])
        print(f"Already downloaded years: {downloaded_years}")
        years_to_download = years_to_download - set(downloaded_years)
    else:
        cur_df = pd.DataFrame()
    
    print(f"Downloading data for the following years: {years_to_download}")

    dfs = [cur_df]
    for year in years_to_download:
        df_avg = download_year(year)
        dfs.append(df_avg)
    df = pd.concat(dfs, ignore_index=True, verify_integrity=True)
    df = df.sort_values(by=['year', 'month', 'day'])
    print(f"Saving daily temperature data to: {savepath}")
    savepath.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(savepath, index=False)


def download_year(year):
    url = URL.format(str(year))
    print(f"Downloading data from: {url}")

    # column names from https://gml.noaa.gov/aftp/data/meteorology/in-situ/README
    # NOTE: this data has NaNs. Only temperature will be handled
    # NOTE: this data has some discrepancy to the temperatures reported on the CALM website
    df = pd.read_csv(url, names=[
        "site_code",
        "year",
        "month",
        "day",
        "hour",
        "wind_direction",
        "wind_speed",
        "wind_steadiness_factor",
        "barometric_pressure",
        "temp_2m_c",
        "temp_10m_c",
        "temp_tt_c",
        "relative_humidity",
        "precipitation_intensity"
    ], delim_whitespace=True)
    
    # ignore other columns
    df = df[['site_code', 'year', 'month', 'day', 'hour', 'temp_2m_c']]
    df['temp_2m_c'].replace(-999.9, np.nan, inplace=True)

    df_brw = df[df['site_code'] == 'BRW']
    df_avg = df_brw.groupby(['site_code', 'year', 'month', 'day']).mean()
    df_avg = df_avg.reset_index().drop('hour', axis=1)
    return df_avg
