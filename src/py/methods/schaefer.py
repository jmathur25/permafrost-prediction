import datetime
import pathlib
import pickle
import click
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

import pandas as pd

import h5py
import tqdm
import sys

sys.path.append("/permafrost-prediction/src/py")
from methods.utils import LatLonFile, compute_stats
from data.consts import CALM_PROCESSSED_DATA_DIR, DATA_PARENT_FOLDER, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR
from data.utils import get_date_for_alos
from methods.soil_models import alt_to_surface_deformation, compute_alt_f_deformation
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    # ("ALPSRP235992170", "ALPSRP242702170"), # processing error
]


@click.command()
def schaefer_method():
    """
    TODO: generalize
    """

    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

    paper_specified_ignore = [7, 110, 121]
    data_specified_ignore = [21, 43, 55]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    calib_point_id = 61
    start_year = 2006
    end_year = 2010

    # If False, ADDT normalized per year. Otherwise, normalized by biggest ADDT across all years.
    norm_per_year = False  # True
    # If False, matrix solves a delta deformation problem with respect to calibration point. A
    # final correction is applied after solving to make the calibration point have the right subsidence.
    # If True, matrix solves a deformation problem. The calibration point is used (with ADDT) to estimate
    # a ground deformation offset that is then applied to make the deformation consistent with the expected deformation.
    correct_E_per_igram = False
    # Run using MintPy instead of Schaefer approach. TODO: split off
    mintpy = True

    multi_threaded = True

    # Use the geo-corrected interferogram products instead of radar geometry
    # TODO: for mintpy and traditional ReSALT, I am not sure this has been implemented correctly.
    use_geo = False

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
            # QUESTION: is "end of season" normalization when barrow measures or just highest
            # DDT per year?
            df_t["norm_ddt"] = df_t["ddt"] / df_t["ddt"].values[-1]
            # date_max_alt = df_peak_alt.loc[calib_point_id, year]["date"]
            # end_of_season_ddt = df_t.loc[year, date_max_alt.month, date_max_alt.day]["ddt"]
            # df_t["norm_ddt"] = df_t["ddt"] / end_of_season_ddt
        df_temps.append(df_t)
    df_temp = pd.concat(df_temps, verify_integrity=True)
    if not norm_per_year:
        max_ddt = df_temp["ddt"].max()
        df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt

    calib_alt = df_alt_gt.loc[calib_point_id]["alt_m"]
    calib_subsidence = alt_to_surface_deformation(calib_alt)
    # print("OVERRIDING SUB")
    # calib_subsidence = 0.0202
    print("CALIBRATION SUBSIDENCE:", calib_subsidence)

    # RHS and LHS per-pixel of eq. 2
    si = SCHAEFER_INTEFEROGRAMS  # SCHAEFER_INTEFEROGRAMS[0:1]
    n = len(si)
    lhs_all = np.zeros((n, len(df_alt_gt)))
    rhs_all = np.zeros((n, 2))
    if mintpy:
        print("Running with MintPy solution")
        correct_E_per_igram = False
        mintpy_output_dir = pathlib.Path("/permafrost-prediction/src/py/methods/mintpy/barrow_2006_2010")
        stack_stripmap_output_dir = DATA_PARENT_FOLDER / "stack_stripmap_outputs/barrow_2006_2010"
        lhs_all, rhs_all = process_mintpy_timeseries(stack_stripmap_output_dir, mintpy_output_dir, df_alt_gt, df_temp, use_geo)
    else:
        if multi_threaded:
            with ThreadPoolExecutor() as executor:
                pbar = tqdm.tqdm(total=n)
                futures = []
                for i, scene_pair in enumerate(si):
                    futures.append(
                        executor.submit(
                            worker,
                            i,
                            scene_pair,
                            df_alt_gt,
                            df_calm,
                            calib_point_id,
                            df_temp,
                            calib_subsidence,
                            correct_E_per_igram,
                            use_geo,
                        )
                    )

                for future in as_completed(futures):
                    pbar.update(1)
                    i, lhs, rhs = future.result()
                    lhs_all[i] = lhs
                    rhs_all[i, :] = rhs
                pbar.close()
        else:
            for i, (scene1, scene2) in enumerate(si):
                lhs, rhs = process_scene_pair(
                    scene1,
                    scene2,
                    df_alt_gt,
                    calib_point_id,
                    df_temp,
                    calib_subsidence,
                    correct_E_per_igram,
                    use_geo,
                )
                lhs_all[i] = lhs
                rhs_all[i, :] = rhs

    print("Solving equations")
    rhs_pi = np.linalg.pinv(rhs_all)
    sol = rhs_pi @ lhs_all

    R = sol[0, :]
    E = sol[1, :]

    if not correct_E_per_igram:
        # TODO: should we scale E based on measurement DDT? Note this also depends if normalization
        # already is per-season with measurement time normalized to 1.
        # # Scale E based on measurement DDT
        # avg_sqrt_ddt = 0.0
        # years = df_temp.index.get_level_values(level=0).unique()
        # for year in years:
        #     # date_max_alt = df_peak_alt.loc[calib_point_id, year]["date"]
        #     # measurement_ddt = df_temp.loc[year, date_max_alt.month, date_max_alt.day]["norm_ddt"]
        #     measurement_ddt = df_temp.loc[year, 12, 31]["norm_ddt"]
        #     avg_sqrt_ddt += np.sqrt(measurement_ddt)
        # avg_sqrt_ddt = avg_sqrt_ddt / len(years)
        # E was formed assuming the measurement occurs at the end of the thaw season, yet
        # in reality it did not. We scale E down by the average sqrt DDT of the measurement
        # across the thaw seasons to get an estimate of the average E at measurement time.
        # TODO: measurement is 1995-2013 average yet this only does years 2006-2010
        # E = E * avg_sqrt_ddt

        idx = np.argwhere(df_alt_gt.index == calib_point_id)[0, 0]
        # E[idx] should be approximately 0
        delta_E = calib_subsidence - E[idx]
        E += delta_E

        print("AVG E", np.mean(E))

    alt_pred = []
    for e, point in zip(E, df_alt_gt.index):
        if e < 1e-3:
            print(f"Skipping {point} due to non-positive deformation")
            alt_pred.append(np.nan)
            continue
        alt = compute_alt_f_deformation(e)
        alt_pred.append(alt)

    alt_pred = np.array(alt_pred)
    alt_gt = df_alt_gt["alt_m"].values
    compute_stats(alt_pred, alt_gt, df_alt_gt.index)

    df_alt_gt.to_csv("df_alt_gt.csv", index=True)
    np.save("subsidence", E)
    np.save("alt_preds", alt_pred)
    np.save("alt_gt", alt_gt)


def worker(i, scene_pair, df_alt_gt, df_calm, calib_point_id, df_temp, calib_subsidence, correct_E_per_igram, use_geo):
    scene1, scene2 = scene_pair
    lhs, rhs = process_scene_pair(
        scene1, scene2, df_alt_gt, calib_point_id, df_temp, calib_subsidence, correct_E_per_igram, use_geo
    )
    return i, lhs, rhs


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


def process_scene_pair(
    alos1, alos2, df_calm_points, calib_point_id, df_temp, calib_subsidence, correct_E_per_igram, use_geo
):
    n_horiz = 1
    n_vert = 1
    if use_geo:
        # Horizontal resolution is around 10m for geo-corrected interferogram and we want 30mx30m
        # TODO: do not hardcode
        n_horiz = 3
    print(f"SPATIAL AVG {n_vert}x{n_horiz}")
    isce_output_dir = ISCE2_OUTPUTS_DIR / f"{alos1}_{alos2}"
    _, alos_d1 = get_date_for_alos(alos1)
    _, alos_d2 = get_date_for_alos(alos2)
    print(f"Processing {alos1} on {alos_d1} and {alos2} on {alos_d2}")
    delta_t_years = (alos_d2 - alos_d1).days / 365
    norm_ddt_d2 = get_norm_ddt(df_temp, alos_d2)
    norm_ddt_d1 = get_norm_ddt(df_temp, alos_d1)
    sqrt_addt_diff = np.sqrt(norm_ddt_d2) - np.sqrt(norm_ddt_d1)
    rhs = [delta_t_years, sqrt_addt_diff]

    intfg_unw_file = isce_output_dir / "interferogram/filt_topophase.unw"
    if use_geo:
        intfg_unw_file = intfg_unw_file.with_suffix(".unw.geo")
    ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    igram_unw_phase = ds.GetRasterBand(2).ReadAsArray()
    # print("USING WRAPPED PHASE")
    # intfg_unw_file = isce_output_dir / 'interferogram/filt_topophase.flat'
    # ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    # igram = ds.GetRasterBand(1).ReadAsArray()
    # print("IGRAM", igram)
    # igram_unw_phase = np.angle(igram)
    if use_geo:
        lat_lon = LatLonFile.XML.create_lat_lon(intfg_unw_file)
    else:
        lat_lon = LatLonFile.RDR.create_lat_lon(isce_output_dir / "geometry")

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

    igram_unw_phase_slice = compute_phase_slice(igram_unw_phase, bbox, point_to_pixel, calib_point_id, n_horiz, n_vert)

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
    n_over_2_vert = n_vert // 2
    extra_vert = n_vert % 2
    n_over_2_horiz = n_horiz // 2
    extra_horiz = n_horiz % 2
    for _, y, x in point_to_pixel:
        y -= bbox[0][0]
        x -= bbox[0][1]
        igram_slice = igram_def[
            y - n_over_2_vert : y + n_over_2_vert + extra_vert, x - n_over_2_horiz : x + n_over_2_horiz + extra_horiz
        ]
        lhs.append(igram_slice.mean())
    return lhs, rhs


def process_mintpy_timeseries(stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, df_temp, use_geo):
    # Two sets of lat/lon, one from geom_reference (which references from radar image),
    # which we use to lookup incidence angle. If `use_geo` is passed, we do the actual
    # reading of the interferogram from the geocoded output, which now presents the data
    # in Earth lat/lon coordinates.
    lat_lon_inc = LatLonFile.RDR.create_lat_lon(stack_stripmap_output_dir / "geom_reference")
    if use_geo:
        lat_lon_intfg = LatLonFile.H5.create_lat_lon(mintpy_outputs_dir / "geo")
    else:
        lat_lon_intfg = lat_lon_inc

    # TODO: can this also be geocoded?
    ds = gdal.Open(str(stack_stripmap_output_dir / "geom_reference/incLocal.rdr"))
    inc = ds.GetRasterBand(2).ReadAsArray()
    del ds
    
    if use_geo:
        f = h5py.File(mintpy_outputs_dir / "geo/geo_timeseries_tropHgt_demErr.h5", "r")
        los_def = f["timeseries"][()]
        dates = f["date"][()]
    else:
        f = h5py.File(mintpy_outputs_dir / "timeseries_tropHgt_demErr.h5", "r")
        los_def = f["timeseries"][()]
        dates = f["date"][()]

    point_to_pixel_inc = []
    point_to_pixel_intfg = []
    for point, row in df_calm_points.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        y, x = lat_lon_inc.find_closest_pixel(lat, lon)
        point_to_pixel_inc.append([point, y, x])

        y, x = lat_lon_intfg.find_closest_pixel(lat, lon)
        point_to_pixel_intfg.append([point, y, x])

    point_to_pixel_inc = np.array(point_to_pixel_inc)
    point_to_pixel_intfg = np.array(point_to_pixel_intfg)

    ground_def = []
    for (py_inc, px_inc), (py_intfg, px_intfg) in zip(point_to_pixel_inc[:,[1,2]], point_to_pixel_intfg[:,[1,2]]):
        incidence_angle = inc[py_inc, px_inc] * np.pi / 180
        ground_def.append(los_def[:, py_intfg, px_intfg] / np.cos(incidence_angle))
    ground_def = np.stack(ground_def).transpose(1, 0)

    lhs = []
    rhs = []
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            alos_d1 = datetime.datetime.strptime(dates[i].decode("utf-8"), "%Y%m%d")
            alos_d2 = datetime.datetime.strptime(dates[j].decode("utf-8"), "%Y%m%d")
            delta_t_years = (alos_d2 - alos_d1).days / 365
            norm_ddt_d2 = get_norm_ddt(df_temp, alos_d2)
            norm_ddt_d1 = get_norm_ddt(df_temp, alos_d1)
            sqrt_addt_diff = np.sqrt(norm_ddt_d2) - np.sqrt(norm_ddt_d1)
            rhs_i = [delta_t_years, sqrt_addt_diff]
            rhs.append(rhs_i)

            pixel_diff = ground_def[j] - ground_def[i]
            lhs.append(pixel_diff)
    lhs = np.stack(lhs)
    rhs = np.array(rhs)
    return lhs, rhs


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


def compute_phase_slice(igram_unw_phase, bbox, point_to_pixel, calib_point_id, n_horiz, n_vert):
    # negative because all inteferograms were computed with granule 1 as the reference
    # and granule 2 as the secondary. This means the phase difference is granule 1 - granule 2.
    # However, the RHS is constructed as data at time of granule 2 - data at time of granule 1.
    # For example, if granule 1 was taken in in June in 2006 and graule 2 in Aug in 2006, we have two things:
    # 1. RHS (time difference and sqrt ADDT diff), which would be those values in Aug - those values in June
    # 2. LHS (delta deformation in Aug - (minus) delta deformation in June). This comes from computing the
    # phase difference, but right now the values flipped. It would give you June - (minus) Aug. Hence we fix that here.
    igram_unw_phase_slice = -igram_unw_phase[bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1]]
    # Use phase-differences wrt to reference pixel
    row = point_to_pixel[point_to_pixel[:, 0] == calib_point_id]
    assert row.shape == (1, 3)
    y = row[0, 1] - bbox[0][0]
    x = row[0, 2] - bbox[0][1]
    n_over_2_vert = n_vert // 2
    extra_vert = n_vert % 2
    n_over_2_horiz = n_horiz // 2
    extra_horiz = n_horiz % 2
    calib_phase_slice = igram_unw_phase_slice[
        y - n_over_2_vert : y + n_over_2_vert + extra_vert, x - n_over_2_horiz : x + n_over_2_horiz + extra_horiz
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
        # diff = ground_def[y, x] + calib_def
        # print(f"Subtracting {diff} from ground deformation to align with calibration deformation")
        # ground_def = ground_def - diff
        ground_def = ground_def + calib_def
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
