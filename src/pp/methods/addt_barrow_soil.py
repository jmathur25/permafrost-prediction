"""
Not relevant to the main paper. Looks at ADDT in Barrow and correlates it with ALT. Might need
some adjusting to work and instructions are likely not complete.
"""

from datetime import datetime
import sys
import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pp.methods.utils import prepare_temp
from pp.data.consts import CALM_PROCESSSED_DATA_DIR, TEMP_DATA_DIR, WORK_DIR

from scipy.stats import pearsonr


@click.command()
def addt_method():
    calm_file = CALM_PROCESSSED_DATA_DIR / "u1/data.csv"

    def extract_month_day(row):
        date = datetime.strptime(f"{row['YEAR']}-{row['DAY']}", "%Y-%j")
        return date.month, date.day

    dfs = []
    norm_per_year = False
    expected_depths_cm = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        45,
        70,
        95,
        120,
    ]
    for i in range(6, 11):
        istr = str(i).zfill(2)
        df = pd.read_excel(WORK_DIR / f"Barrow1_{istr}ave.xls")
        col_renames = dict()
        for i, ed in zip(range(2, len(expected_depths_cm) + 2), expected_depths_cm):
            col = f"MRC.{i}"
            assert df[col].values[0] == f"{ed} cm"
            col_renames[col] = ed
        df = df.dropna(subset="YEAR")
        df = df.rename(col_renames, axis=1)
        df["YEAR"] = df["YEAR"].astype(int)
        df["DAY"] = df["DAY"].astype(int)
        df["month"], df["day"] = zip(*df.apply(extract_month_day, axis=1))
        df = df.rename({"YEAR": "year"}, axis=1)
        cols = ["year", "month", "day"] + list(col_renames.values())
        dfs.append(df[cols])
    df_soil_temp = pd.concat(dfs)
    # df_temp = df_temp.set_index(['year', 'month', 'day'])
    del df

    # Calculate the depth of the 0-degree isotherm per row
    def zero_isotherm_depth(row):
        freezing_point = 0
        depths = expected_depths_cm
        if row[depths[0]] < freezing_point:
            return 0  # Some number less than 5 cm. TODO: We could linearly interpolate to surface temp??
        for i in range(len(depths) - 1):
            # If a temperature cross is detected (one temp is positive and the other is negative)
            if row[depths[i]] > freezing_point and row[depths[i + 1]] < freezing_point:
                # Linear interpolation to find the depth of 0-degree isotherm
                slope = (row[depths[i + 1]] - row[depths[i]]) / (
                    depths[i + 1] - depths[i]
                )
                zero_crossing_depth = (
                    depths[i] + (freezing_point - row[depths[i]]) / slope
                )
                assert depths[i] < zero_crossing_depth < depths[i + 1]
                return zero_crossing_depth
        return depths[-1]

    df_soil_temp["0_degree_depth"] = df_soil_temp.apply(zero_isotherm_depth, axis=1)

    temp_file = TEMP_DATA_DIR / "barrow/data/data.csv"
    start_year = 2006
    end_year = 2010
    norm_per_year = False
    df_air_temp = prepare_temp(temp_file, start_year, end_year, norm_per_year)

    df_soil_temp = pd.merge(
        df_soil_temp, df_air_temp, on=["year", "month", "day"], how="left"
    )
    del df_air_temp

    def print_stats(x, y, x_desc, y_desc):
        pearson_r, _ = pearsonr(x, y)
        print(f"{x_desc} vs {y_desc} R:", pearson_r)

        plt.scatter(x, y)
        plt.text(
            0.05,
            0.95,
            f"Pearson R: {pearson_r:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="gray", facecolor="aliceblue"
            ),
        )
        plt.ylabel(y_desc)
        plt.xlabel(x_desc)
        sp = f"addt_corr_{x_desc}_vs_{y_desc}.png"
        print("Saving to", sp)
        plt.savefig(sp)
        plt.close()

    do_month = 6
    df_soil_temp = df_soil_temp[df_soil_temp["month"] == do_month]

    y = df_soil_temp["0_degree_depth"].values
    x = df_soil_temp["ddt"].values
    print_stats(x, y, f"ddt_on_month_{do_month}", "0_degree_depth")
    x = np.sqrt(df_soil_temp["ddt"].values)
    print_stats(x, y, f"sqrt_ddt_on_month_{do_month}", "0_degree_depth")

    # paper_specified_ignore = [7, 110, 121]
    # data_specified_ignore = [21, 43, 55]
    # ignore_point_ids = paper_specified_ignore + data_specified_ignore
    # start_year = 2006
    # end_year = 2010

    # ddt_scale = False
    # df_calm = load_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale)
    # df_calm = pd.merge(df_calm, df_temp, on=['year', 'month', 'day'], how='left')

    print()


if __name__ == "__main__":
    addt_method()
