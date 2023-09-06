import pickle
import click
import numpy as np
import matplotlib.pyplot as plt

import gdal

import pandas as pd

import tqdm
from methods.utils import compute_stats, LatLon
from data.consts import CALM_PROCESSSED_DATA_DIR, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR
from data.utils import get_date_for_alos
from methods.soil_models import alt_to_surface_deformation, compute_alt_f_deformation

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
    ("ALPSRP074952170", "ALPSRP189022170"),  # fix
    ("ALPSRP074952170", "ALPSRP235992170"),
    ("ALPSRP081662170", "ALPSRP128632170"),
    ("ALPSRP081662170", "ALPSRP182312170"),
    # ("ALPSRP081662170", "ALPSRP128632170"), # dup 10
    ("ALPSRP081662170", "ALPSRP189022170"),  # fix
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

    ignore_point_ids = [7, 21, 43, 55, 110, 121]
    calib_point_id = 61
    start_year = 2006
    end_year = 2010

    # If False, ADDT normalized per year. Otherwise, normalized by biggest ADDT across all years.
    norm_per_year = True
    # If False, matrix solves a delta deformation problem with respect to calibration point. A
    # final correction is applied after tje
    # If True, matrix solves a deformation problem. The calibration point is used (with ADDT) to estimate
    # a ground deformation offset that is then applied to make the deformation consistent with the expected deformation.
    correct_E_per_igram = False

    df_calm = pd.read_csv(calm_file, parse_dates=["date"])
    df_calm = df_calm.sort_values("date", ascending=True)
    df_calm = df_calm[df_calm["point_id"].apply(lambda x: x not in ignore_point_ids)]
    df_calm = df_calm[(df_calm["date"] >= pd.to_datetime("1995")) & (df_calm["date"] < pd.to_datetime("2014"))]

    def try_float(x):
        try:
            return float(x)
        except:
            return np.nan

    # TODO: fix in processor. handle 'w'?
    df_calm["alt_m"] = df_calm["alt_m"].apply(try_float) / 100
    # only grab ALTs from end of summer, which willl be the last
    # measurement in a year
    df_calm["year"] = df_calm["date"].dt.year
    df_peak_alt = df_calm.groupby(["point_id", "year"]).last()
    df_alt_gt = df_peak_alt.groupby("point_id").mean()
    df_alt_gt = df_alt_gt.drop("date", axis=1)

    df_temp = pd.read_csv(temp_file)
    assert len(pd.unique(df_temp["site_code"])) == 1  # TODO: support codes
    df_temp = df_temp[(df_temp["year"] >= start_year) & (df_temp["year"] < end_year + 1)]
    df_temp = df_temp.sort_values(["year", "month", "day"]).set_index(["year", "month", "day"])
    df_temps = []
    for year, df_t in df_temp.groupby("year"):
        compute_ddt_ddf(df_t)
        if norm_per_year:
            # df_t["norm_ddt"] = df_t["ddt"] / df_t["ddt"].values[-1]
            date_max_alt = df_peak_alt.loc[calib_point_id, year]["date"]
            end_of_season_ddt = df_t.loc[year, date_max_alt.month, date_max_alt.day]["ddt"]
            df_t["norm_ddt"] = df_t["ddt"] / end_of_season_ddt
        df_temps.append(df_t)
    df_temp = pd.concat(df_temps, verify_integrity=True)
    if not norm_per_year:
        max_ddt = df_temp["ddt"].max()
        df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt

    calib_alt = df_alt_gt.loc[calib_point_id]["alt_m"]
    calib_subsidence = alt_to_surface_deformation(calib_alt)
    # print("OVERRIDING SUB")
    # calib_subsidence = alt_to_surface_deformation(calib_alt)
    print("CALIBRATION SUBSIDENCE:", calib_subsidence)

    # RHS and LHS per-pixel of eq. 2
    si = SCHAEFER_INTEFEROGRAMS  # SCHAEFER_INTEFEROGRAMS[0:1]
    n = len(si)
    lhs_all = np.zeros((n, len(df_alt_gt)))
    rhs_all = np.zeros((n, 2))
    for i, (scene1, scene2) in enumerate(si):
        lhs, rhs, point_to_pixel = process_scene_pair(
            scene1, scene2, df_alt_gt, calib_point_id, df_temp, calib_subsidence, correct_E_per_igram
        )
        lhs_all[i] = lhs
        rhs_all[i, :] = rhs

    print("Solving equations")
    rhs_pi = np.linalg.pinv(rhs_all)
    sol = rhs_pi @ lhs_all

    R = sol[0, :]
    E = sol[1, :]

    np.save("R", R)
    np.save("E", E)

    if not correct_E_per_igram:
        E = E + calib_subsidence
        # df_maxes = df_temp.groupby("year").max()
        # E_sol = E_sol * np.mean(np.sqrt(df_maxes["norm_ddt"].values))

    alt_pred = []
    # TODO: point to pixel needs to be represented better.
    # we are assuming `process_scene_pair` keeps them
    # in the same order.
    for e, point in zip(E, point_to_pixel):
        if e < 1e-3:
            print(f"Skipping {point} due to non-positive deformation")
            alt_pred.append(np.nan)
            continue
        alt = compute_alt_f_deformation(e)
        alt_pred.append(alt)

    alt_pred = np.array(alt_pred)
    alt_gt = df_alt_gt["alt_m"].values
    compute_stats(alt_pred, alt_gt)


def compute_ddt_ddf(df):
    tmp_col = "temp_2m_c"
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

    df["ddf"] = ddf_list
    df["ddt"] = ddt_list


def process_scene_pair(alos1, alos2, df_calm_points, calib_point_id, df_temp, calib_subsidence, correct_E_per_igram):
    n = 1
    print(f"SPATIAL AVG n={n}")
    isce_output_dir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"
    alos_d1 = get_date_for_alos(alos1)
    alos_d2 = get_date_for_alos(alos2)
    print(f"Processing {alos1} on {alos_d1} and {alos2} on {alos_d2}")
    delta_t_years = (alos_d2 - alos_d1).days / 365
    norm_ddt_d2 = get_norm_ddt(df_temp, alos_d2)
    print("NORM DT AFTER", norm_ddt_d2)
    norm_ddt_d1 = get_norm_ddt(df_temp, alos_d1)
    print("NORM DT BEFORE", norm_ddt_d1)
    sqrt_addt_diff = np.sqrt(norm_ddt_d2) - np.sqrt(norm_ddt_d1)
    rhs = [delta_t_years, sqrt_addt_diff]

    intfg_unw_file = isce_output_dir / "interferogram/filt_topophase.unw"
    ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    igram_unw_phase = ds.GetRasterBand(2).ReadAsArray()
    # print("USING WRAPPED PHASE")
    # intfg_unw_file = isce_output_dir / 'interferogram/filt_topophase.flat'
    # ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    # igram = ds.GetRasterBand(1).ReadAsArray()
    # print("IGRAM", igram)
    # igram_unw_phase = np.angle(igram)
    lat_lon = LatLon(isce_output_dir)

    with open(isce_output_dir / "PICKLE/interferogram", "rb") as fp:
        pickle_isce_obj = pickle.load(fp)
    radar_wavelength = pickle_isce_obj["reference"]["instrument"]["radar_wavelength"]
    incidence_angle = pickle_isce_obj["reference"]["instrument"]["incidence_angle"] * np.pi / 180

    print("radar wavelength:", radar_wavelength)
    print("incidence angle:", incidence_angle)

    point_to_pixel = []
    for point, row in df_calm_points.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        y, x = lat_lon.find_closest_pixel(lat, lon)
        point_to_pixel.append([point, y, x])
    point_to_pixel = np.array(point_to_pixel)

    bbox = compute_bounding_box(point_to_pixel[:, [1, 2]])
    print(f"Bounding box set to: {bbox}")

    igram_unw_phase_slice = compute_phase_slice(
        igram_unw_phase, bbox, point_to_pixel, calib_point_id, correct_E_per_igram, n=n
    )
    scaled_calib_def = calib_subsidence * sqrt_addt_diff
    igram_def = compute_deformation(
        igram_unw_phase_slice,
        bbox,
        incidence_angle,
        radar_wavelength,
        point_to_pixel,
        calib_point_id,
        scaled_calib_def,
        correct_E_per_igram,
    )

    lhs = []
    n_over_2 = n // 2
    extra = n % 2
    for _, y, x in point_to_pixel:
        y -= bbox[0][0]
        x -= bbox[0][1]
        igram_slice = igram_def[y - n_over_2 : y + n_over_2 + extra, x - n_over_2 : x + n_over_2 + extra]
        lhs.append(igram_slice.mean())
    return lhs, rhs, point_to_pixel


def get_norm_ddt(df_temp, date):
    return df_temp.loc[date.year, date.month, date.day]["norm_ddt"]


def get_ddt(df_temp, date):
    return df_temp.loc[date.year, date.month, date.day]["ddt"]


def plot_change(img, bbox, point_to_pixel, label):
    plt.imshow(img, cmap="viridis", origin="lower")

    # Add red boxes
    for point in point_to_pixel:
        point_id, y, x = point
        y -= bbox[0][0]
        x -= bbox[0][1]
        plt.gca().add_patch(plt.Rectangle((x - 1.5, y - 1.5), 3, 3, fill=None, edgecolor="red", linewidth=2))

        # Annotate each box with the point #
        plt.annotate(
            f"#{point_id}", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=5, color="white"
        )

    plt.colorbar()
    plt.title(label)
    plt.show()


def compute_phase_slice(igram_unw_phase, bbox, point_to_pixel, calib_point_id, correct_E_per_igram, n):
    igram_unw_phase_slice = -igram_unw_phase[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]
    if not correct_E_per_igram:
        # Use phase-differences wrt to reference pixel
        row = point_to_pixel[point_to_pixel[:, 0] == calib_point_id]
        assert row.shape == (1, 3)
        y = row[0, 1] - bbox[0][0]
        x = row[0, 2] - bbox[0][1]
        n_over_2 = n // 2
        extra = n % 2
        calib_phase_slice = igram_unw_phase_slice[
            y - n_over_2 : y + n_over_2 + extra, x - n_over_2 : x + n_over_2 + extra
        ]
        igram_unw_phase_slice = igram_unw_phase_slice - calib_phase_slice.mean()
    return igram_unw_phase_slice


def compute_deformation(
    igram_unw_delta_phase_slice,
    bbox,
    incidence_angle,
    wavelength,
    point_to_pixel,
    calib_point_id,
    calib_def,
    correct_E_per_igram,
):
    # TODO: incident angle per pixel
    los_def = igram_unw_delta_phase_slice / (2 * 2 * np.pi) * wavelength
    ground_def = los_def / np.cos(incidence_angle)
    row = point_to_pixel[point_to_pixel[:, 0] == calib_point_id]
    assert row.shape == (1, 3)
    y = row[0, 1] - bbox[0][0]
    x = row[0, 2] - bbox[0][1]
    if correct_E_per_igram:
        diff = ground_def[y, x] - calib_def
        print(f"Subtracting {diff} from ground deformation to align with calibration deformation")
        ground_def = ground_def - diff
    return ground_def


def compute_bounding_box(pixels, n=10):
    # Initialize min and max coordinates for y and x
    min_y = np.min(pixels[:, 0])
    min_x = np.min(pixels[:, 1])
    max_y = np.max(pixels[:, 0])
    max_x = np.max(pixels[:, 1])

    # Add 50-pixel margin to each side
    min_y = max(min_y - n, 0)
    min_x = max(min_x - n, 0)
    max_y += n
    max_x += n

    return ((min_y, min_x), (max_y, max_x))


if __name__ == "__main__":
    schaefer_method()
