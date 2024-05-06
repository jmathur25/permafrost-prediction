"""
Get data from
https://daac.ornl.gov/ABOVE/guides/ReSALT_InSAR_Barrow.html#HDataDescrAccess
and place at: /permafrost-prediction/work/barrow_2015/
The folder should look like:
/permafrost-prediction/barrow_2015/
  comp/
  data/
  guide/
"""

# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("/permafrost-prediction/src/")
from pp.methods.soil_models import LiuSMM
from pp.data.consts import CALM_PROCESSSED_DATA_DIR, TEMP_DATA_DIR, WORK_DIR
from pp.methods.utils import compute_stats, prepare_calm_data, prepare_temp


# %%
df = pd.read_csv(WORK_DIR / "barrow_2015/comp/ReSALT_barrow.txt", sep=';')
df['geometry'] = df['WKT'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# %%
df.head()

# %%
gdf.plot()

# %%
calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"
start_year = 1995
end_year = 2013

paper_specified_ignore = [7, 110, 121]
data_specified_ignore = [23, 45, 57]
ignore_point_ids = paper_specified_ignore + data_specified_ignore
# %%
df_temp = prepare_temp(temp_file, start_year, end_year)
df_peak_alt = prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, df_temp)

# Stores information on the avg root DDT at measurement time across the years
# TODO: technically this assumes each point was measured at the same time, which currently is true.
df_avg_measurement_alt_sqrt_ddt = df_peak_alt[['point_id', 'sqrt_norm_ddt']].groupby(['point_id']).mean()
df_peak_alt = df_peak_alt.drop(['year', 'month', 'day', 'norm_ddt'], axis=1) # not needed anymore
df_alt_gt = df_peak_alt.groupby("point_id").mean()

# %%
def find_match(row):
    point = Point(row['longitude'], row['latitude']) # 71.309183
    mask = gdf.contains(point)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        # print(f"Got no matches for {row}")
        # return np.nan
        # Above commented code is what I did before I knew which points were excluded. The paper
        # said 6 points were excluded but only told us about 3. Once I figured out the 3, I
        # made `data_specified_ignore` and made this an error here.
        raise ValueError()
    elif len(indices) == 1:
        return indices[0][0]
    else:
        raise ValueError()
    

matched_indices = df_alt_gt.apply(find_match, axis=1)

# %%
df_alt_gt['matched_indices'] = matched_indices 
df_alt_merged = pd.merge(df_alt_gt, gdf, left_on='matched_indices', right_index=True)

# %%
is_ok = df_alt_merged.apply(lambda row: row['geometry'].contains(Point(row['longitude'], row['latitude'])), axis=1)
assert is_ok.values.all()

# %%
df_alt_merged['alt_pred'] = df_alt_merged['ReSALT_(cm)']/100
df_alt_merged['sub_pred'] = df_alt_merged['Sub_(cm)']/100


# %%
# Reproduces paper!
calib_point_id = 61
can_use_mask = df_alt_merged.index!=calib_point_id
alt_pred = df_alt_merged['alt_pred'].values[can_use_mask]
alt_gt = df_alt_merged['alt_m'].values[can_use_mask]

compute_stats(alt_pred, alt_gt)

# %%
# Mostly confirms Liu SMM aligns with the one used in Schaefer et al. (2015)
smm = LiuSMM()
alts = []
for sub in df_alt_merged['sub_pred']:
    alt = smm.thaw_depth_from_deformation(sub)
    alts.append(alt)
alts = np.array(alts)

line_x = np.linspace(0.2, 0.7)
line_y = line_x

plt.scatter(alts, df_alt_merged['alt_pred'], color='b')
plt.scatter(line_x, line_y, color='r')
plt.xlabel('alt_smm')
plt.ylabel('alt_pred')

print("RMSE LiuSMM:", np.sqrt(np.mean(np.square(alts-df_alt_merged['alt_pred'].values))))

# %%
