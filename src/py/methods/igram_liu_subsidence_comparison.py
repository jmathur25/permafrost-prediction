# %%
%load_ext autoreload
%autoreload 2

# %%
import pathlib
import sys
from matplotlib import pyplot as plt

import pandas as pd
import tqdm
from scipy.stats import pearsonr

sys.path.append("..")
from data.utils import get_date_for_alos
from data.consts import WORK_FOLDER, ISCE2_OUTPUTS_DIR
from methods.igrams import SCHAEFER_INTEFEROGRAMS, get_mintpy_deformation_timeseries, plot_change, process_igram

# %%
df_liu_sub_gt = pd.read_csv("/permafrost-prediction-shared-data/Liu-Larson_2018.tab", delimiter="\t", skiprows=16, parse_dates=['Date/Time'])
df_liu_sub_gt['year'] = df_liu_sub_gt['Date/Time'].dt.year
df_liu_sub_gt['month'] = df_liu_sub_gt['Date/Time'].dt.month
df_liu_sub_gt['day'] = df_liu_sub_gt['Date/Time'].dt.day
df_liu_sub_gt = df_liu_sub_gt.set_index(['year', 'month', 'day'])

df_liu_sub_gt['Elev change [m]'] = df_liu_sub_gt['Elev change [cm]'] / 100

# %%
def get_ec(alos_date, df, allow_nearest):
    try:
        row = df.loc[alos_date.year, alos_date.month, alos_date.day]
    except:
        if allow_nearest:
            assert alos_date.month < 7
            days = df.loc[alos_date.year, 7]
            return days['Elev change [m]'].values[0]
        else:
            return None
    return row['Elev change [m]']

# %%
use_mintpy = True

usable_igrams = []
if use_mintpy:
    usable_igrams = SCHAEFER_INTEFEROGRAMS
else:
    allow_nearest = True
    for (alos1, alos2) in SCHAEFER_INTEFEROGRAMS:
        _, alos_d1 = get_date_for_alos(alos1)
        _, alos_d2 = get_date_for_alos(alos2)
        if alos_d1.month == alos_d2.month:
            continue
        ec1 = get_ec(alos_d1, df_liu_sub_gt, allow_nearest)
        ec2 = get_ec(alos_d2, df_liu_sub_gt, allow_nearest)
        if allow_nearest:
            assert ec1 is not None and ec2 is not None
        if ec1 is not None and ec2 is not None:
            usable_igrams.append((alos1, alos2))

    skip_inds = [4, 9]
    print("Ignoring", skip_inds)
    use_inds = sorted(list(set(list(range(len(usable_igrams)))) - set(skip_inds)))

    usable_igrams = [usable_igrams[i] for i in use_inds]
    
print(len(usable_igrams))

# %%
df_gt_sub_locs = pd.DataFrame.from_records({
    'point_id': [0],
    'latitude': [71.323000],
    'longitude': [-156.610000],
})
n_horiz = 1
n_vert = 1
use_geo = False

sub_expecteds = []
sub_actuals = []
if use_mintpy:
    print("Running with MintPy")
    mintpy_output_dir = pathlib.Path("/permafrost-prediction/src/py/methods/mintpy/barrow_2006_2010")
    stack_stripmap_output_dir = WORK_FOLDER / "stack_stripmap_outputs/barrow_2006_2010"
    dates, ground_def = get_mintpy_deformation_timeseries(stack_stripmap_output_dir, mintpy_output_dir, df_gt_sub_locs, use_geo)
    for i in range(1, len(dates)):
        alos_d1 = dates[i - 1]
        alos_d2 = dates[i]
        sub_expected = get_ec(alos_d2, df_liu_sub_gt, allow_nearest) - get_ec(alos_d1, df_liu_sub_gt, allow_nearest)
        sub_actual = ground_def[i,0] - ground_def[i - 1,0]

        sub_expecteds.append(sub_expected)
        sub_actuals.append(sub_actual)
else:
    for idx in tqdm.tqdm(range(len(usable_igrams))):
        alos1 = usable_igrams[idx][0]
        alos2 = usable_igrams[idx][1]
        isce_output_dir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"

        _, alos_d1 = get_date_for_alos(alos1)
        _, alos_d2 = get_date_for_alos(alos2)
        print("Looking at", alos_d1, alos_d2)

        point_to_pixel, bbox, igram_def, lat_lon = process_igram(df_gt_sub_locs, None, use_geo, n_horiz, n_vert, isce_output_dir)

        sub_expected = get_ec(alos_d2, df_liu_sub_gt, allow_nearest) - get_ec(alos_d1, df_liu_sub_gt, allow_nearest)
        sub_actual = igram_def[point_to_pixel[0,1] - bbox[0][0], point_to_pixel[0,2] - bbox[0][1]]
        sub_expecteds.append(sub_expected)
        sub_actuals.append(sub_actual)


# %%
pearson_r, _ = pearsonr(sub_expecteds, sub_actuals)
print("Pearson R", pearson_r)
plt.scatter(sub_expecteds, sub_actuals)
plt.xlabel('Expected subsidence')
plt.ylabel('Actual subsidence')

# %%
