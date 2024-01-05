# %%
%load_ext autoreload
%autoreload 2

# %%
import datetime
import netCDF4 as nc
import sys

import pandas as pd
import numpy as np

sys.path.append("/permafrost-prediction/src/py")
from methods.resalt import ReSALT, ReSALT_Type
from methods.soil_models import ChenSMM, LiuSMM
from methods.utils import LatLonArray, compute_ddt_ddf, compute_stats, prepare_calm_data, prepare_temp
from data.consts import CALM_PROCESSSED_DATA_DIR, DATA_PARENT_FOLDER, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR

# %%
ds = nc.Dataset(
    "/permafrost-prediction-shared-data/pdo/PDO_ReSALT_barrow_2017_03.nc4",
    "r",
)
# %%
lat = ds['lat'][:]
lon = ds['lon'][:]
alt = ds['alt'][:]
sub = ds['sub'][:]

# %%
lat_lon = LatLonArray(lat, lon)

# %%
ds_temp = nc.Dataset('/permafrost-prediction-shared-data/temperature/daymet/daymet_v4_daily_na_tmax_2017.nc', 'r')

# %%
lat_dm = ds_temp['lat'][:]
lon_dm = ds_temp['lon'][:]
# temp = ds_temp['tmax'][:]
lat_lon_dm = LatLonArray(lat_dm.data, lon_dm.data)

# %%
y, x = lat_lon_dm.find_closest_pixel(71.310720, -156.589301, max_dist_meters=501)
print(y, x)

# %%
dates = pd.date_range(start='2017-01-01', end='2017-12-31')

df = pd.DataFrame({
    'date': dates,
    'temp_2m_c': ds_temp['tmax'][:, y, x]
})

# %%
# Extract year, month, and day and add them as separate columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop the original Date column if it's no longer needed
df.drop('date', axis=1, inplace=True)

# %%
calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
# temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

start_year = 2017
end_year = 2017
ignore_point_ids = []
ddt_scale = False

# df_temp = prepare_temp(temp_file, start_year, end_year)
df = df.set_index(["year", "month", "day"])
df_temps = []
for year, df_t in df.groupby("year"):
    compute_ddt_ddf(df_t)
    df_temps.append(df_t)
df_temp = pd.concat(df_temps, verify_integrity=True)
max_ddt = df_temp["ddt"].max()
df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt

df_peak_alt = prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp)
# Stores information on the avg root DDT at measurement time across the years
# TODO: technically this assumes each point was measured at the same time, which currently is true.
df_avg_measurement_alt_sqrt_ddt = df_peak_alt[['point_id', 'sqrt_norm_ddt']].groupby(['point_id']).mean()
df_peak_alt = df_peak_alt.drop(['year', 'month', 'day', 'norm_ddt'], axis=1) # not needed anymore
df_alt_gt = df_peak_alt.groupby("point_id").mean()

# %%
sub_per_pix = []
for _, r in df_alt_gt.iterrows():
    lat = r['latitude']
    lon = r['longitude']
    y, x = lat_lon.find_closest_pixel(lat, lon)
    sub_ = sub.data[y, x]
    if sub_ == -9999.0:
        sub_ = np.nan
    sub_per_pix.append(sub_)
    
# %%
dates = [
    (datetime.datetime(2017, 9, 16), datetime.datetime(2017, 6, 21))
]

sub_per_pix = np.array(sub_per_pix)
nan_mask = np.isnan(sub_per_pix)
sub_per_pix_not_nan = sub_per_pix[~nan_mask]
sub_per_pix_not_nan = sub_per_pix_not_nan[None, :]

# %%
smm = ChenSMM()
rtype = ReSALT_Type.LIU_SCHAEFER
resalt = ReSALT(df_temp, smm, None, None, rtype)
alt_pred = resalt.run_inversion(sub_per_pix_not_nan, dates, only_solve_E=True)

# %%
df_avg_measurement_alt_sqrt_ddt = df_peak_alt[['point_id', 'sqrt_norm_ddt']].groupby(['point_id']).mean()
alt_pred = alt_pred * df_avg_measurement_alt_sqrt_ddt['sqrt_norm_ddt'].values[~nan_mask]

# %%
alt_gt = df_alt_gt['alt_m'].values[~nan_mask]
compute_stats(alt_pred, alt_gt)

# %%


# %%

