import pickle
import click
from isce2.components import isceobj
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from scipy.integrate import quad
from scipy.optimize import root_scalar

import gdal

import pandas as pd

import tqdm
from methods.utils import compute_stats, load_img, LatLon
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR
from data.utils import get_date_for_alos

SCHAEFER_INTEFEROGRAMS = [
    ("ALPSRP021272170", "ALPSRP027982170"),
    ("ALPSRP021272170", "ALPSRP128632170"),
    ("ALPSRP021272170", "ALPSRP182312170"),
    ("ALPSRP021272170", "ALPSRP189022170"),
    ("ALPSRP027982170", "ALPSRP182312170"),
    ("ALPSRP074952170", "ALPSRP081662170"),
    ("ALPSRP074952170", "ALPSRP128632170"),
    ("ALPSRP074952170", "ALPSRP182312170"),
    # ("ALPSRP074952170", "ALPSRP128632170"), # dup 6
    ("ALPSRP074952170", "ALPSRP235992170"),
    ("ALPSRP081662170", "ALPSRP128632170"),
    ("ALPSRP081662170", "ALPSRP182312170"),
    # ("ALPSRP081662170", "ALPSRP128632170"), # dup 10
    ("ALPSRP081662170", "ALPSRP189022170"),
    ("ALPSRP081662170", "ALPSRP242702170"),
    ("ALPSRP128632170", "ALPSRP182312170"),
    ("ALPSRP128632170", "ALPSRP189022170"),
    ("ALPSRP182312170", "ALPSRP189022170"),
    ("ALPSRP189022170", "ALPSRP235992170"),
    # gdal fails? ERROR 4: `/permafrost-prediction/src/py/data/isce2_outputs/ALPSRP235992170_ALPSRP242702170/interferogram/filt_topophase.unw' not recognized as a supported file format.
    # ("ALPSRP235992170", "ALPSRP242702170"),
]


@click.command()
def schaefer_method():
    """
        TODO: generalize
    """
    
    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"
    
    ignore_point_ids = [
        7,
        110,
        121
    ]
    calib_point_id = 61
    start_year = 2006
    end_year = 2010

    df_temp = pd.read_csv(temp_file)
    assert len(pd.unique(df_temp['site_code'])) == 1  # TODO: support codes
    df_temp = df_temp[(df_temp['year'] >= start_year) & (df_temp['year'] <= end_year)]
    df_temp = df_temp.sort_values(['year', 'month', 'day'])
    df_temps = []
    for (_, df_t) in df_temp.groupby(['year']):
        compute_ddt_ddf(df_t)
        df_t['norm_ddt'] = df_t['ddt'] / df_t['ddt'].values[-1]
        df_temps.append(df_t)
    df_temp = pd.concat(df_temps, verify_integrity=True)
    # max_ddt = df_temp['ddt'].max()
    # df_temp['norm_ddt'] = df_temp['ddt'] / max_ddt
    df_temp = df_temp.set_index(['year', 'month', 'day'])
    
    df_calm = pd.read_csv(calm_file, parse_dates=['date'])
    df_calm = df_calm[df_calm['point_id'].apply(lambda x: x not in ignore_point_ids)]
    df_calm = df_calm[(df_calm['date'] >= pd.to_datetime("2006")) & (df_calm['date'] <= pd.to_datetime("2010"))]
    def try_float(x):
        try:
            return float(x)
        except:
            return np.nan
    # TODO: fix in processor. handle 'w'?
    df_calm['alt_m'] = df_calm['alt_m'].apply(try_float) / 100
    df_peak_alt = df_calm[df_calm['date'].dt.month.isin([8, 9])]
    df_alt_gt = df_peak_alt.groupby('point_id').mean()
    df_calm_points = df_calm.drop_duplicates(subset=['point_id'])

    calib_alt = df_alt_gt.loc[calib_point_id]['alt_m']
    calib_subsidence = alt_to_surface_deformation(calib_alt)
    # calib_subsidence = 0.012991343843446251

    # RHS and LHS per-pixel of eq. 2
    si = SCHAEFER_INTEFEROGRAMS # [("ALPSRP182312170", "ALPSRP189022170")] # SCHAEFER_INTEFEROGRAMS[5:6]
    n = len(SCHAEFER_INTEFEROGRAMS)
    lhs_all = np.zeros((n, len(df_calm_points)))
    rhs_all = np.zeros((n, 2))
    for i, (scene1, scene2) in enumerate(si[:n]):
        lhs, rhs, point_to_pixel = process_scene_pair(scene1, scene2, df_calm_points, calib_point_id, df_temp, calib_subsidence)
        lhs_all[i] = lhs
        rhs_all[i] = rhs
    
    print("Solving equations")
    rhs_all2 = rhs_all[:, 1:2]
    rhs_pi = np.linalg.pinv(rhs_all)
    rhs_pi2 = np.linalg.pinv(rhs_all2)
    sol = rhs_pi @ lhs_all
    sol2 = rhs_pi2 @ lhs_all
    
    R = sol[0, :]
    E = sol[1, :]
    # E = sol[0, :]
    
    np.save("R", R)
    np.save("E", E)
    
    for E_sol, m in zip([E, sol2[0]], ["R and E", "just E"]):
        print(f"----\nAnalyzing method {m}\n----")
        
        E_sol = E_sol + calib_subsidence

        alt_pred = []
        # TODO: point to pixel needs to be represented better. 
        # we are assuming `process_scene_pair` keeps them
        # in the same order.
        for e, point in zip(E_sol, point_to_pixel):
            if e < 0:
                print(f"Skipping {point} due to neg deformation")
                alt_pred.append(np.nan)
                continue
            alt = compute_alt_f_deformation(e)
            alt_pred.append(alt)
        
        alt_pred = np.array(alt_pred)
        alt_gt = df_alt_gt['alt_m'].values
        compute_stats(alt_pred, alt_gt)
    
    # print("CHECKING SIMPLER PRED")
    # new_pred_def = lhs_all + calib_subsidence
    # assert new_pred_def.shape[0] == 1
    # new_pred_def = new_pred_def[0]
    # new_pred_alt = np.array([compute_alt_f_deformation(e) if e > 0 else np.nan for e in new_pred_def])
    # print("num nans", np.isnan(new_pred_alt).sum())
    # compute_stats(new_pred_alt, alt_gt)
    
    import code
    code.interact(local=locals())

def compute_ddt_ddf(df):
    tmp_col = 'temp_2m_c'
    freezing_point = 0.0
    # Initialize counters
    ddf = 0
    ddt = 0
    ddf_list = []
    ddt_list = []

    # Iterate over the temp_2m_c column
    for i, temp in enumerate(df[tmp_col]):
        if not np.isnan(temp):
            if temp < freezing_point:
                ddf += temp
            else:
                ddt += temp
        ddf_list.append(ddf)
        ddt_list.append(ddt)

    df['ddf'] = ddf_list
    df['ddt'] = ddt_list
    

def process_scene_pair(alos1, alos2, df_calm_points, calib_point_id, df_temp, calib_subsidence):
    n = 1 # spatial avg procedures use a 3x3 window
    isce_output_dir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"
    alos_d1 = get_date_for_alos(alos1)
    alos_d2 = get_date_for_alos(alos2)
    print(f"Processing {alos1} on {alos_d1} and {alos2} on {alos_d2}")
    delta_t_years = (alos_d2 - alos_d1).days / 365
    norm_ddt_d2 = get_norm_ddt(df_temp, alos_d2)
    print("NORM DT AFTER", norm_ddt_d2)
    norm_ddt_d1 = get_norm_ddt(df_temp, alos_d1)
    print("NORM DT BEFORE", norm_ddt_d1)
    # ddt_d2 = get_ddt(df_temp, alos_d2)
    # ddt_d1 = get_ddt(df_temp, alos_d1)
    # larger = max(ddt_d2, ddt_d1)
    # print("OVERRIDING DDT")
    # norm_ddt_d2 = ddt_d2 / larger
    # norm_ddt_d1 = ddt_d1 / larger
    # print("NORM DTs", norm_ddt_d1, norm_ddt_d2)
    sqrt_addt_diff = np.sqrt(norm_ddt_d2) - np.sqrt(norm_ddt_d1)
    rhs = [delta_t_years, sqrt_addt_diff]
        
    intfg_unw_file = isce_output_dir / 'interferogram/filt_topophase.unw'
    ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    igram_unw_phase = ds.GetRasterBand(2).ReadAsArray()
    lat_lon = LatLon(isce_output_dir)
    
    with open(isce_output_dir / "PICKLE/interferogram", "rb") as fp:
        pickle_isce_obj = pickle.load(fp)
    radar_wavelength = pickle_isce_obj['reference']['instrument']['radar_wavelength']
    incidence_angle = pickle_isce_obj['reference']['instrument']['incidence_angle'] * np.pi/180

    print("radar wavelength:", radar_wavelength)
    print("incidence angle:", incidence_angle)

    point_to_pixel = []
    for (i, row) in tqdm.tqdm(df_calm_points.iterrows(), total=len(df_calm_points)):
        point = row['point_id']
        lat = row['latitude']
        lon = row['longitude']
        y, x = lat_lon.find_closest_pixel(lat, lon)
        point_to_pixel.append([point, y, x])
    point_to_pixel = np.array(point_to_pixel)
    
    bbox = compute_bounding_box(point_to_pixel[:,[1,2]])
    print(f"Bounding box set to: {bbox}")
    
    igram_unw_delta_phase_slice = compute_delta_phase_slice(igram_unw_phase, bbox, point_to_pixel, calib_point_id, n=n)
    igram_delta_def = compute_delta_deformation(igram_unw_delta_phase_slice, bbox, incidence_angle, radar_wavelength)

    lhs = []
    to_add = 0  #calib_subsidence if alos_d2 > alos_d1 else -calib_subsidence
    n_over_2 = n // 2
    extra = n % 2
    for (_, y, x) in point_to_pixel:
        y -= bbox[0][0]
        x -= bbox[0][1]
        igram_slice = igram_delta_def[y - n_over_2:y + n_over_2 + extra, x - n_over_2 : x + n_over_2 + extra]
        lhs.append(igram_slice.mean() + to_add)
    return lhs, rhs, point_to_pixel
        
        
def get_norm_ddt(df_temp, date):
    return df_temp.loc[date.year, date.month, date.day]['norm_ddt']


def get_ddt(df_temp, date):
    return df_temp.loc[date.year, date.month, date.day]['ddt']


def plot_change(img, bbox, point_to_pixel, label):
    plt.imshow(img, cmap='viridis', origin='lower')

    # Add red boxes
    for point in point_to_pixel:
        point_id, y, x = point
        y -= bbox[0][0]
        x -= bbox[0][1]
        plt.gca().add_patch(plt.Rectangle((x - 1.5, y - 1.5), 3, 3, fill=None, edgecolor='red', linewidth=2))
        
        # Annotate each box with the point #
        plt.annotate(f"#{point_id}", (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=5, color='white')

    plt.colorbar()
    plt.title(label)
    plt.show()

def compute_delta_phase_slice(igram_unw_phase, bbox, point_to_pixel, calib_point_id, n):
    igram_unw_phase_slice = -igram_unw_phase[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]
    row = point_to_pixel[point_to_pixel[:,0] == calib_point_id]
    assert row.shape == (1, 3)
    y = row[0, 1] - bbox[0][0]
    x = row[0, 2] - bbox[0][1]
    n_over_2 = n//2
    extra = n % 2
    calib_phase_slice = igram_unw_phase_slice[y - n_over_2:y+n_over_2 + extra, x-n_over_2:x+n_over_2+extra]
    return igram_unw_phase_slice - calib_phase_slice.mean()

def compute_delta_deformation(igram_unw_delta_phase_slice, bbox, incidence_angle, wavelength):
    # TODO: incident angle per pixel
    los_def = igram_unw_delta_phase_slice/(2*np.pi)*wavelength
    ground_def = los_def / np.cos(incidence_angle)
    return ground_def

def compute_bounding_box(pixels, n=10):
    # Initialize min and max coordinates for y and x
    min_y = np.min(pixels[:,0])
    min_x = np.min(pixels[:,1])
    max_y = np.max(pixels[:,0])
    max_x = np.max(pixels[:,1])

    # Add 50-pixel margin to each side
    min_y = max(min_y - n, 0)
    min_x = max(min_x - n, 0)
    max_y += n
    max_x += n
    
    return ((min_y, min_x), (max_y, max_x))

def resalt_integrand(z):
    po = 0.9
    pf = 0.45
    # Chosen to make the VWC ~0.7 for an ALT of 0.36 cm
    # because paper reported this general average and this ALT average
    # for CALM
    k = 4
    return pf + (po - pf)*np.exp(-k*z)


def alt_to_surface_deformation(alt):
    # paper assumes exponential decay from 90% porosity to 45% porosity
    # in general:
    # P(z) = P_f + (Po - Pf)*e^(-kz)
    # where P(f) = final porosity
    #       P(o) = intial porosity
    #       z is rate of exponential decay
    # Without reading citation, let us assume k = 1
    # Definite integral is now: https://www.wolframalpha.com/input?i=integrate+a+%2B+be%5E%28-kz%29+dz+from+0+to+x
    pw = 0.997 # g/m^3
    pi = 0.9168 # g/cm^3
    integral, error = quad(resalt_integrand, 0, alt)
    print("INTEGRAL/ALT", integral/alt)
    assert error < 1e-5
    return (pw - pi) / pi * integral


def compute_alt_f_deformation(deformation):
    assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
    po = 0.9
    pf = 0.45
    k = 1
    pw = 0.997 # g/m^3
    pi = 0.9168 # g/cm^3
    integral_val = deformation * (pi / (pw - pi))
    
    # Define the function to find its root
    def objective(x, target):
        integral, _ = quad(resalt_integrand, 0, x)
        return integral - target

    result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method='brentq')
    assert result.converged
    return result.root

if __name__ == '__main__':
    schaefer_method()
