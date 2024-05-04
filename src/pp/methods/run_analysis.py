"""
Main file for reproducing the paper's results.
"""

import datetime
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal


import h5py
import tqdm

from pp.methods.resalt import ReSALT, ReSALT_Type
from pp.methods.igrams import SCHAEFER_INTEFEROGRAMS
from pp.methods.utils import (
    LatLonFile,
    compute_stats,
    get_norm_ddt,
    prepare_calm_data,
    prepare_temp,
)
from pp.data.consts import (
    CALM_PROCESSSED_DATA_DIR,
    ISCE2_OUTPUTS_DIR,
    TEMP_DATA_DIR,
    WORK_DIR,
)
from pp.data.utils import get_date_for_alos
from pp.methods.soil_models import LiuSMM
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_analysis():
    """
    This runs the analysis. Read the code to see what to change to run other types of analysis.
    """

    # -- CONFIG --

    # The type of algorithm
    rtype = ReSALT_Type.SCReSALT

    # Give the title and savepath of where to save results
    # ("SCReSALT Results All Data", pathlib.Path('sc_resalt_results_full.png'))
    plot = None

    # These are specific to Schaefer et al. (2015)
    paper_specified_ignore = [7, 110, 121]
    # These are not explicitly provided in Schaefer et al. (2015) but inferred in `benchmark_resalt_barrow_2015.py`
    data_specified_ignore = [23, 45, 57]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    # The calibration node to calibrate subsidences in InSAR images
    calib_point_id = 61

    # Temperature and ALT data will be grabbed from these years
    start_year = 1995
    end_year = 2013

    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

    interferograms = SCHAEFER_INTEFEROGRAMS

    multi_threaded = True

    # Use the geo-corrected interferogram products instead of radar geometry. Improves results.
    use_geo = True

    # Use MintPy's time-series approach to correcting deformations
    # TODO: does not currently work and is not well-documented
    use_mintpy = False

    # -- END CONFIG --

    df_temp = prepare_temp(temp_file, start_year, end_year)
    df_peak_alt = prepare_calm_data(
        calm_file, ignore_point_ids, start_year, end_year, df_temp
    )
    # Stores information on the avg root DDT at measurement time across the years
    df_avg_measurement_alt_ddt = (
        df_peak_alt[["point_id", "norm_ddt"]].groupby(["point_id"]).mean()
    )
    df_peak_alt = df_peak_alt.drop(
        ["year", "month", "day", "norm_ddt"], axis=1
    )  # not needed anymore
    df_alt_gt = df_peak_alt.groupby("point_id").mean()

    calib_alt = df_alt_gt.loc[calib_point_id]["alt_m"]
    calib_ddt = df_avg_measurement_alt_ddt.loc[calib_point_id]['norm_ddt']

    # TODO: now invalid...
    # The calibration ALT needs to be upscaled to the end-of-season thaw depth. The ADDT at
    # end-of-season is 1.0 because ADDT is normalized. Hence, by using Stefan scaling, we can
    # use the the end-of-season ADDT, the average sqrt ADDT at measurement time, and the average
    # ALT at measurement time to upscale to the average end-of-season thaw depth.
    # upscale = (
    #     1.0
    #     / df_avg_measurement_alt_sqrt_ddt.loc[calib_point_id]["norm_ddt"].mean()
    # )
    # calib_alt = calib_alt * upscale

    liu_smm = LiuSMM()
    calib_subsidence = liu_smm.deformation_from_alt(calib_alt)
    matches = np.argwhere(df_alt_gt.index == calib_point_id)
    assert matches.shape == (1, 1)
    calib_idx = matches[0, 0]
    print("Calibration subsidence:", calib_subsidence)

    resalt = ReSALT(df_temp, liu_smm, calib_idx, calib_subsidence, calib_ddt, rtype)

    if use_mintpy:
        print("RUNNING USING MINTPY")
        mintpy_output_dir = pathlib.Path(
            "/permafrost-prediction/src/pp/methods/mintpy/barrow_2006_2010"
        )
        stack_stripmap_output_dir = WORK_DIR / "stack_stripmap_outputs/barrow_2006_2010"
        # Specifically give MintPy just the point id, lat, lon to reduce any chance of data leakage.
        df_point_locs = df_alt_gt[["latitude", "longitude"]]
        deformations, dates = process_mintpy_timeseries(
            stack_stripmap_output_dir, mintpy_output_dir, df_point_locs, use_geo
        )
    else:
        n = len(interferograms)
        deformations = np.zeros((n, df_alt_gt.shape[0]))
        dates = [None] * n
        if multi_threaded:
            with ThreadPoolExecutor() as executor:
                pbar = tqdm.tqdm(total=n)
                futures = []
                for i, scene_pair in enumerate(interferograms):
                    futures.append(
                        executor.submit(
                            worker,
                            i,
                            scene_pair,
                            df_alt_gt,
                            use_geo,
                        )
                    )

                for future in as_completed(futures):
                    pbar.update(1)
                    i, deformation, date_pair = future.result()
                    deformations[i, :] = deformation
                    dates[i] = date_pair
                pbar.close()
        else:
            for i, (scene1, scene2) in enumerate(interferograms):
                deformation, date_pair = process_scene_pair(
                    scene1, scene2, df_alt_gt, use_geo, True
                )
                deformations[i, :] = deformation
                dates[i] = date_pair

    alt_pred = resalt.run_inversion(deformations, dates)
    # Scale to measurement time
    # TODO:...
    #alt_pred = alt_pred * df_avg_measurement_alt_ddt["sqrt_norm_ddt"].values

    alt_gt = df_alt_gt["alt_m"].values

    # Sanity check
    err = abs(alt_gt[calib_idx] - alt_pred[calib_idx])
    assert err < 5e-3

    # Remove calibration point from ALTs
    can_use_mask = df_alt_gt.index != calib_point_id
    alt_gt = alt_gt[can_use_mask]
    alt_pred = alt_pred[can_use_mask]

    compute_stats(alt_pred, alt_gt, plot=plot)

    plt.scatter(alt_gt, alt_pred)
    plt.xlabel("ALT Ground-Truth")
    plt.ylabel("ALT Prediction")


def worker(i, scene_pair, df_alt_gt, use_geo):
    scene1, scene2 = scene_pair
    deformation, date_pair = process_scene_pair(
        scene1, scene2, df_alt_gt, use_geo, True
    )
    return i, deformation, date_pair


def process_scene_pair(alos1, alos2, df_calm_points, use_geo, needs_sign_flip):
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
    if needs_sign_flip:
        alos_d1, alos_d2 = alos_d2, alos_d1
    point_to_pixel, igram_def, _ = process_igram(
        df_calm_points, use_geo, n_horiz, n_vert, isce_output_dir, needs_sign_flip
    )
    return igram_def, (alos_d1, alos_d2)


def process_igram(
    df_calm_points, use_geo, n_horiz, n_vert, isce_output_dir, needs_sign_flip
):
    intfg_unw_file = isce_output_dir / "interferogram/filt_topophase.unw"
    if use_geo:
        intfg_unw_file = intfg_unw_file.with_suffix(".unw.geo")
    ds = gdal.Open(str(intfg_unw_file), gdal.GA_ReadOnly)
    igram_unw_phase_img = ds.GetRasterBand(2).ReadAsArray()

    if use_geo:
        lat_lon = LatLonFile.XML.create_lat_lon(
            intfg_unw_file, check_dims=igram_unw_phase_img.shape
        )
    else:
        lat_lon = LatLonFile.RDR.create_lat_lon(
            isce_output_dir / "geometry", check_dims=igram_unw_phase_img.shape
        )

    with open(isce_output_dir / "PICKLE/interferogram", "rb") as fp:
        pickle_isce_obj = pickle.load(fp)
    radar_wavelength = pickle_isce_obj["reference"]["instrument"]["radar_wavelength"]
    incidence_angle = (
        pickle_isce_obj["reference"]["instrument"]["incidence_angle"] * np.pi / 180
    )
    # Using inc_explore.py, figured this out. Incidence angle mostly seems the same and is close
    # to the extracted number, so I am not including parsing it in this analysis. I'm not even
    # sure how to get this value from ISCE2. Mintpy gets it using the stack stripmap processor.
    # print("OVERRIDING INCIDENCE ANGLE")
    # incidence_angle = 0.701683

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

    igram_unw_phase = compute_phase_slice(
        igram_unw_phase_img, point_to_pixel, n_horiz, n_vert, needs_sign_flip
    )
    igram_def = compute_deformation(
        igram_unw_phase,
        incidence_angle,
        radar_wavelength,
    )
    return point_to_pixel, igram_def, lat_lon


def process_mintpy_timeseries(
    stack_stripmap_output_dir,
    mintpy_outputs_dir,
    df_point_locs,
    use_geo,
):
    dates, ground_def = get_mintpy_deformation_timeseries(
        stack_stripmap_output_dir, mintpy_outputs_dir, df_point_locs, use_geo
    )
    deformation_per_pixel = []
    igram_dates = []
    # We do all (n C 2) combinations of dates.
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            alos_d1 = dates[i]
            alos_d2 = dates[j]
            deformation_per_pixel.append(ground_def[j] - ground_def[i])
            igram_dates.append((alos_d1, alos_d2))
    deformation_per_pixel = np.stack(deformation_per_pixel)
    return deformation_per_pixel, igram_dates


def get_mintpy_deformation_timeseries(
    stack_stripmap_output_dir, mintpy_outputs_dir, df_point_locs, use_geo
):
    # Two sets of lat/lon, one from geom_reference (which references from radar image),
    # which we use to lookup incidence angle. If `use_geo` is passed, we do the actual
    # reading of the interferogram from the geocoded output, which now presents the data
    # in Earth lat/lon coordinates.
    lat_lon_inc = LatLonFile.RDR.create_lat_lon(
        stack_stripmap_output_dir / "geom_reference"
    )
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

    dates = [datetime.datetime.strptime(d.decode("utf-8"), "%Y%m%d") for d in dates]

    point_to_pixel_inc = []
    point_to_pixel_intfg = []
    for point, row in df_point_locs.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        # TODO: not sure why max dist is a bit higher for these...
        y, x = lat_lon_inc.find_closest_pixel(lat, lon, max_dist_meters=50)
        point_to_pixel_inc.append([point, y, x])

        y, x = lat_lon_intfg.find_closest_pixel(lat, lon, max_dist_meters=60)
        point_to_pixel_intfg.append([point, y, x])

    point_to_pixel_inc = np.array(point_to_pixel_inc)
    point_to_pixel_intfg = np.array(point_to_pixel_intfg)

    ground_def = []
    for (py_inc, px_inc), (py_intfg, px_intfg) in zip(
        point_to_pixel_inc[:, [1, 2]], point_to_pixel_intfg[:, [1, 2]]
    ):
        incidence_angle = inc[py_inc, px_inc] * np.pi / 180
        ground_def.append(los_def[:, py_intfg, px_intfg] / np.cos(incidence_angle))
    ground_def = np.stack(ground_def).transpose(1, 0)

    return dates, ground_def


def get_ddt(df_temp, date):
    return df_temp.loc[date.year, date.month, date.day]["ddt"]


def plot_change(img, bbox, point_to_pixel, label):
    plt.imshow(img, cmap="viridis", origin="lower")

    # Add red boxes
    for point in point_to_pixel:
        point_id, y, x = point
        y -= bbox[0][0]
        x -= bbox[0][1]
        plt.gca().add_patch(
            plt.Rectangle(
                (x - 1.5, y - 1.5), 3, 3, fill=None, edgecolor="red", linewidth=2
            )
        )

        # Annotate each box with the point #
        plt.annotate(
            f"#{point_id}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=5,
            color="white",
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
    y_start, y_end = max(0, y - half_vert), min(
        igram.shape[0], y + half_vert + extra_vert
    )
    x_start, x_end = max(0, x - half_horiz), min(
        igram.shape[1], x + half_horiz + extra_horiz
    )

    # Extract the box and compute the average
    box = igram[y_start:y_end, x_start:x_end]
    return np.mean(box)


def compute_phase_slice(
    igram_unw_phase_img, point_to_pixel, n_horiz, n_vert, needs_sign_flip
):
    # grab the phases of the study points
    igram_unw_phase = np.array(
        [
            extract_average(igram_unw_phase_img, y, x, n_horiz, n_vert)
            for _, y, x in point_to_pixel
        ]
    )

    if needs_sign_flip:
        # negative because all inteferograms were computed with granule 1 as the reference
        # and granule 2 as the secondary. This means the phase difference is granule 1 - granule 2.
        # However, the RHS is constructed as data at time of granule 2 - data at time of granule 1.
        # For example, if granule 1 was taken in in June in 2006 and graule 2 in Aug in 2006, we have two things:
        # 1. RHS (time difference and sqrt ADDT diff), which would be those values in Aug - those values in June
        # 2. LHS (delta deformation in Aug - (minus) delta deformation in June). This comes from computing the
        # phase difference, but right now the values flipped. It would give you June - (minus) Aug. Hence we fix that here.
        igram_unw_phase = -igram_unw_phase
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


if __name__ == "__main__":
    run_analysis()
