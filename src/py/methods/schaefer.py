import datetime
import pathlib
import pickle
from typing import Optional, Union
import click
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

import pandas as pd

import h5py
import tqdm
import sys



# TODO: fix module-ing
sys.path.append("/permafrost-prediction/src/py")
from methods.igrams import SCHAEFER_INTEFEROGRAMS
from methods.utils import LatLonFile, compute_stats, prepare_calm_data, prepare_temp
from methods.gt_phase_unwrap import solve_best_phase_unwrap
from data.consts import CALM_PROCESSSED_DATA_DIR, DATA_PARENT_FOLDER, ISCE2_OUTPUTS_DIR, TEMP_DATA_DIR
from data.utils import get_date_for_alos
from methods.soil_models import LiuSMM
from concurrent.futures import ThreadPoolExecutor, as_completed

@click.command()
def schaefer_method():
    """
    TODO: generalize
    """

    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

    paper_specified_ignore = [7, 110, 121]
    data_specified_ignore = [21, 43, 55] + [8, 9, 20, 34, 45, 56, 67, 68, 78, 89, 90, 100, 101, 102, 111]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    calib_point_id = 61
    start_year = 2006
    end_year = 2010

    # If False, ADDT normalized per year. Otherwise, normalized by biggest ADDT across all years.
    norm_per_year = False  # True
    
    # Run using MintPy instead of Schaefer approach. TODO: split off
    mintpy = False
    
    # If True, uses Roger/Chen redefined way to compute ADDT diff
    sqrt_ddt_correction = False

    multi_threaded = True

    # Use the geo-corrected interferogram products instead of radar geometry
    # TODO: for mintpy and traditional ReSALT, I am not sure this has been implemented correctly.
    use_geo = False

    # Scale measured and estimated ALTs by the DDT at measure time (probe or SAR) to be the one at end-of-season when thaw is maximized
    # TODO: should scaling subsidence be independent of scaling ALT? Technically we could just scale deformations
    # to what they should be at measurement time.
    ddt_scale = False

    df_temp = prepare_temp(temp_file, start_year, end_year)
    df_peak_alt = prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp)
    df_peak_alt = df_peak_alt.drop(['year', 'month', 'day', 'norm_ddt'], axis=1) # not needed anymore
    df_alt_gt = df_peak_alt.groupby("point_id").mean()

    calib_alt = df_alt_gt.loc[calib_point_id]["alt_m"]
    liu_smm = LiuSMM()
    calib_subsidence = liu_smm.deformation_from_alt(calib_alt)
    # print("OVERRIDING SUB")
    # calib_subsidence = 0.0202
    print("CALIBRATION SUBSIDENCE:", calib_subsidence)

    # RHS and LHS per-pixel of eq. 2
    si = SCHAEFER_INTEFEROGRAMS
    n = len(si)
    lhs_all = np.zeros((n, len(df_alt_gt)))
    rhs_all = np.zeros((n, 2))
    if mintpy:
        print("Running with MintPy solution")
        mintpy_output_dir = pathlib.Path("/permafrost-prediction/src/py/methods/mintpy/barrow_2006_2010")
        stack_stripmap_output_dir = DATA_PARENT_FOLDER / "stack_stripmap_outputs/barrow_2006_2010"
        lhs_all, rhs_all = process_mintpy_timeseries(stack_stripmap_output_dir, mintpy_output_dir, df_alt_gt, df_temp, use_geo, sqrt_ddt_correction)
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
                            calib_point_id,
                            df_temp,
                            use_geo,
                            sqrt_ddt_correction
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
                    use_geo,
                    sqrt_ddt_correction,
                    True
                )
                lhs_all[i] = lhs
                rhs_all[i, :] = rhs

    print("Solving equations")
    # rhs_all = rhs_all[:, ]
    rhs_pi = np.linalg.pinv(rhs_all)
    sol = rhs_pi @ lhs_all

    # R = sol[0, :]
    E = sol[1, :]
    # E = sol[0, :]

    # # TODO: should we scale E based on measurement DDT? Note this also depends if normalization
    # # already is per-season with measurement time normalized to 1, and `ddt_scale`.
    # # Scale E based on measurement DDT
    # avg_sqrt_ddt = 0.0
    # years = df_temp.index.get_level_values(level=0).unique()
    # for year in years:
    #     # date_max_alt = df_peak_alt.loc[calib_point_id, year]["date"]
    #     # measurement_ddt = df_temp.loc[year, date_max_alt.month, date_max_alt.day]["norm_ddt"]
    #     measurement_ddt = df_temp.loc[year, 12, 31]["norm_ddt"]
    #     avg_sqrt_ddt += np.sqrt(measurement_ddt)
    # avg_sqrt_ddt = avg_sqrt_ddt / len(years)
    # # E was formed assuming the measurement occurs at the end of the thaw season, yet
    # # in reality it did not. We scale E down by the average sqrt DDT of the measurement
    # # across the thaw seasons to get an estimate of the average E at measurement time.
    # # TODO: measurement is 1995-2013 average yet this only does years 2006-2010
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
        alt = liu_smm.alt_from_deformation(e)
        alt_pred.append(alt)

    alt_pred = np.array(alt_pred)
    
    alt_gt = df_alt_gt["alt_m"].values
    compute_stats(alt_pred, alt_gt)

    # df_alt_gt.to_csv("df_alt_gt.csv", index=True)
    # np.save("subsidence", E)
    # np.save("alt_preds", alt_pred)
    # np.save("alt_gt", alt_gt)


def worker(i, scene_pair, df_alt_gt, calib_point_id, df_temp, use_geo, sqrt_ddt_correction):
    scene1, scene2 = scene_pair
    lhs, rhs = process_scene_pair(
        scene1, scene2, df_alt_gt, calib_point_id, df_temp, use_geo, sqrt_ddt_correction, True
    )
    return i, lhs, rhs


def process_scene_pair(
    alos1, alos2, df_calm_points, calib_point_id, df_temp, use_geo, sqrt_ddt_correction, needs_sign_flip, ideal_deformation: Optional[np.ndarray]=None
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
    delta_t_years = (alos_d1 - alos_d2).days / 365
    norm_ddt_d1 = get_norm_ddt(df_temp, alos_d1)
    norm_ddt_d2 = get_norm_ddt(df_temp, alos_d2)
    if sqrt_ddt_correction:
        diff = norm_ddt_d1 - norm_ddt_d2
        if diff > 0:
            sqrt_addt_diff = np.sqrt(diff)
        else:
            sqrt_addt_diff = -np.sqrt(-diff)
    else:
        sqrt_addt_diff = np.sqrt(norm_ddt_d1) - np.sqrt(norm_ddt_d2)
    rhs = [delta_t_years, sqrt_addt_diff]
    if needs_sign_flip:
        # If the signs were inverted in processing, then d2 is the reference not d1.
        # TODO: make the arguments just reference, secondary?
        rhs = [-ri for ri in rhs]

    point_to_pixel, igram_def, _ = process_igram(df_calm_points, calib_point_id, use_geo, n_horiz, n_vert, isce_output_dir, needs_sign_flip, ideal_deformation)
    return igram_def, rhs

def process_igram(df_calm_points, calib_point_id, use_geo, n_horiz, n_vert, isce_output_dir, needs_sign_flip, ideal_deformation: Optional[np.ndarray]=None):
    intfg_unw_file = isce_output_dir / "interferogram/filt_topophase.unw"
    if use_geo:
        intfg_unw_file = intfg_unw_file.with_suffix(".unw.geo")
    ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    igram_unw_phase_img = ds.GetRasterBand(2).ReadAsArray()
    
    if use_geo:
        lat_lon = LatLonFile.XML.create_lat_lon(intfg_unw_file, check_dims=igram_unw_phase_img.shape)
    else:
        lat_lon = LatLonFile.RDR.create_lat_lon(isce_output_dir / "geometry", check_dims=igram_unw_phase_img.shape)

    with open(isce_output_dir / "PICKLE/interferogram", "rb") as fp:
        pickle_isce_obj = pickle.load(fp)
    radar_wavelength = pickle_isce_obj["reference"]["instrument"]["radar_wavelength"]
    incidence_angle = pickle_isce_obj["reference"]["instrument"]["incidence_angle"] * np.pi / 180

    print("radar wavelength (meteres):", radar_wavelength)
    print("incidence angle (radians):", incidence_angle)

    point_to_pixel = []
    for point, row in df_calm_points.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        y, x = lat_lon.find_closest_pixel(lat, lon)
        point_to_pixel.append([point, y, x])
    point_to_pixel = np.array(point_to_pixel)

    # bbox = compute_bounding_box(point_to_pixel[:, [1, 2]])
    # print(f"Bounding box set to: {bbox}")
    
    igram_unw_phase = compute_phase_slice(igram_unw_phase_img, point_to_pixel, calib_point_id, n_horiz, n_vert, needs_sign_flip)
    
    if ideal_deformation is not None:
        print("Correcting for phase unwrapping errors using ideal deformation")
        # TODO: per pixel incidence angle??
        ideal_unw_phase = ideal_deformation_to_phase(ideal_deformation, incidence_angle, radar_wavelength)
        igram_wrapped_phase = igram_unw_phase - np.round(igram_unw_phase / (2 * np.pi)) * 2 * np.pi
        igram_unw_phase = solve_best_phase_unwrap(ideal_unw_phase, igram_wrapped_phase)
    
    igram_def = compute_deformation(
        igram_unw_phase,
        incidence_angle,
        radar_wavelength,
    )
    
    return point_to_pixel,igram_def,lat_lon


def process_mintpy_timeseries(stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, df_temp, use_geo, sqrt_ddt_correction):
    dates, ground_def = get_mintpy_deformation_timeseries(stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, use_geo)
    lhs = []
    rhs = []
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            alos_d1 = dates[i]
            alos_d2 = dates[j]
            delta_t_years = (alos_d2 - alos_d1).days / 365
            norm_ddt_d2 = get_norm_ddt(df_temp, alos_d2)
            norm_ddt_d1 = get_norm_ddt(df_temp, alos_d1)
            if sqrt_ddt_correction:
                diff = norm_ddt_d2 - norm_ddt_d1
                if diff > 0:
                    sqrt_addt_diff = np.sqrt(diff)
                else:
                    sqrt_addt_diff = -np.sqrt(-diff)
            else:
                sqrt_addt_diff = np.sqrt(norm_ddt_d2) - np.sqrt(norm_ddt_d1)
            rhs_i = [delta_t_years, sqrt_addt_diff]
            rhs.append(rhs_i)

            pixel_diff = ground_def[j] - ground_def[i]
            lhs.append(pixel_diff)
    lhs = np.stack(lhs)
    rhs = np.array(rhs)
    return lhs, rhs


def get_mintpy_deformation_timeseries(stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, use_geo):
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
        
    dates = [datetime.datetime.strptime(d.decode("utf-8"), "%Y%m%d")
            for d in dates]

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
    
    return dates, ground_def


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


def extract_average(igram, y, x, n_horiz, n_vert):
    half_horiz = n_horiz // 2
    half_vert = n_vert // 2
    extra_horiz = n_horiz % 2
    extra_vert = n_vert % 2
    
    # Define the bounding box coordinates
    y_start, y_end = max(0, y - half_vert), min(igram.shape[0], y + half_vert + extra_vert)
    x_start, x_end = max(0, x - half_horiz), min(igram.shape[1], x + half_horiz + extra_horiz)

    # Extract the box and compute the average
    box = igram[y_start:y_end, x_start:x_end]
    return np.mean(box)


def compute_phase_slice(igram_unw_phase_img, point_to_pixel, calib_point_id, n_horiz, n_vert, needs_sign_flip):
    # grab the phases of the study points
    igram_unw_phase = np.array([extract_average(igram_unw_phase_img, y, x, n_horiz, n_vert) for _, y, x in point_to_pixel])

    if needs_sign_flip:
        # negative because all inteferograms were computed with granule 1 as the reference
        # and granule 2 as the secondary. This means the phase difference is granule 1 - granule 2.
        # However, the RHS is constructed as data at time of granule 2 - data at time of granule 1.
        # For example, if granule 1 was taken in in June in 2006 and graule 2 in Aug in 2006, we have two things:
        # 1. RHS (time difference and sqrt ADDT diff), which would be those values in Aug - those values in June
        # 2. LHS (delta deformation in Aug - (minus) delta deformation in June). This comes from computing the
        # phase difference, but right now the values flipped. It would give you June - (minus) Aug. Hence we fix that here.
        igram_unw_phase = -igram_unw_phase
    if calib_point_id is not None:
        # Use phase-differences wrt to reference pixel
        idx = np.argwhere(point_to_pixel[:, 0] == calib_point_id)
        assert idx.shape == (1,1)
        ref_phase = igram_unw_phase[idx[0,0]]
        igram_unw_phase = igram_unw_phase - ref_phase
    return igram_unw_phase


def compute_deformation(
    igram_unw_delta_phase_slice,
    incidence_angle,
    wavelength,
):
    # TODO: incident angle per pixel
    los_def = igram_unw_delta_phase_slice / (4 * np.pi) * wavelength
    ground_def = los_def / np.cos(incidence_angle)
    return ground_def


def ideal_deformation_to_phase(ideal_deformation: np.array, incidence_angle: float, wavelength: float) -> np.ndarray:
    ideal_los_def = ideal_deformation * np.cos(incidence_angle)
    ideal_igram_unw_phase = ideal_los_def / wavelength * (4 * np.pi)
    return ideal_igram_unw_phase
    

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
