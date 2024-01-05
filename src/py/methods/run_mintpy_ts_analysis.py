import datetime
import pathlib
import pickle
from typing import Optional, Union
import click
import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

import h5py
import sys


# TODO: fix module-ing
sys.path.append("/permafrost-prediction/src/py")
from methods.utils import (
    LatLonFile,
    compute_stats,
    get_norm_ddt,
    prepare_calm_data,
    prepare_temp,
)
from data.consts import CALM_PROCESSSED_DATA_DIR, WORK_FOLDER, TEMP_DATA_DIR
from methods.soil_models import LiuSMM


@click.command()
def mintpy_method():
    """
    TODO: generalize
    """

    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"
    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"

    paper_specified_ignore = [7, 110, 121]
    data_specified_ignore = [21, 43, 55] + [
        8,
        9,
        20,
        34,
        45,
        56,
        67,
        68,
        78,
        89,
        90,
        100,
        101,
        102,
        111,
    ]
    ignore_point_ids = paper_specified_ignore + data_specified_ignore
    calib_point_id = 61
    start_year = 2006
    end_year = 2010

    # Use the geo-corrected interferogram products instead of radar geometry
    # TODO: for mintpy and traditional ReSALT, I am not sure this has been implemented correctly.
    use_geo = False

    # Scale measured and estimated ALTs by the DDT at measure time (probe or SAR) to be the one at end-of-season when thaw is maximized
    # TODO: should scaling subsidence be independent of scaling ALT? Technically we could just scale deformations
    # to what they should be at measurement time.
    ddt_scale = False

    df_temp = prepare_temp(temp_file, start_year, end_year)
    df_peak_alt = prepare_calm_data(
        calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp
    )
    df_peak_alt = df_peak_alt.drop(
        ["year", "month", "day", "norm_ddt"], axis=1
    )  # not needed anymore
    df_alt_gt = df_peak_alt.groupby("point_id").mean()

    calib_alt = df_alt_gt.loc[calib_point_id]["alt_m"]
    liu_smm = LiuSMM()
    calib_subsidence = liu_smm.deformation_from_alt(calib_alt)
    # print("OVERRIDING SUB")
    # calib_subsidence = 0.0202
    print("CALIBRATION SUBSIDENCE:", calib_subsidence)

    # RHS and LHS per-pixel of eq. 2
    print("Running with MintPy solution")
    mintpy_output_dir = pathlib.Path(
        "/permafrost-prediction/src/py/methods/mintpy/barrow_2006_2010"
    )
    stack_stripmap_output_dir = WORK_FOLDER / "stack_stripmap_outputs/barrow_2006_2010"
    lhs_all, rhs_all = process_mintpy_timeseries(
        stack_stripmap_output_dir, mintpy_output_dir, df_alt_gt, df_temp, use_geo
    )

    print("Solving equations")
    # rhs_all = rhs_all[:, ]
    rhs_pi = np.linalg.pinv(rhs_all)
    sol = rhs_pi @ lhs_all

    # R = sol[0, :]
    E = sol[1, :]
    # E = sol[0, :]

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


def process_mintpy_timeseries(
    stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, df_temp, use_geo
):
    dates, ground_def = get_mintpy_deformation_timeseries(
        stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, use_geo
    )
    lhs = []
    rhs = []
    for i in range(len(dates)):
        for j in range(i + 1, len(dates)):
            alos_d1 = dates[i]
            alos_d2 = dates[j]
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


def get_mintpy_deformation_timeseries(
    stack_stripmap_output_dir, mintpy_outputs_dir, df_calm_points, use_geo
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
    for point, row in df_calm_points.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        y, x = lat_lon_inc.find_closest_pixel(lat, lon, max_dist_meters=50)
        point_to_pixel_inc.append([point, y, x])

        y, x = lat_lon_intfg.find_closest_pixel(lat, lon, max_dist_meters=50)
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


if __name__ == "__main__":
    mintpy_method()
