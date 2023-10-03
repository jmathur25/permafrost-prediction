import datetime
import enum
import pathlib
from typing import Optional
import click
import tqdm
from urllib.parse import urljoin
import pandas as pd
import requests
import code

from data.consts import DATA_PARENT_FOLDER, CALM_PROCESSSED_DATA_DIR, CALM_RAW_DATA_DIR
from data.sar import alos_palsar_granule
from data.utils import prompt_user


class StandardDataFormatColumns(enum.Enum):
    # The ordering here will be the ordering of headers
    DATE = "date"
    POINT_ID = "point_id"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    ALT_METERS = "alt_m"

    @classmethod
    def save_standardized_dataframe(cls, df: pd.DataFrame, site_code: str):
        try:
            cls.check_cols(df)
        except Exception as e:
            raise ValueError(f"Invalid df columns: {df.columns}. Error: {e}")
        savepath = CALM_PROCESSSED_DATA_DIR / site_code / "data.csv"
        savepath.parent.mkdir(parents=True, exist_ok=True)
        # Technically we should do this check before running all the processing code. Re-evaluate later.
        if savepath.exists():
            if not prompt_user(f"Data already processed to {savepath}. Reprocess?"):
                return

        # order the columns in a consistent order
        df = df[[c.value for c in StandardDataFormatColumns]]
        df.to_csv(savepath, index=False)

    @classmethod
    def check_cols(cls, df):
        assert len(df.columns) == len(cls)
        for c in cls:
            assert c.value in df.columns


class CALMDownloadSite(enum.Enum):
    BARROW = "Barrow"

    def download(self):
        def download_file(url: str, savepath: pathlib.Path):
            savepath.parent.mkdir(parents=True, exist_ok=True)
            if savepath.exists():
                if not prompt_user(f"Data already downloaded at {savepath}. Redownload?"):
                    return
            resp = requests.get(url)
            if resp.status_code == 200:
                with open(savepath, "wb") as f:
                    f.write(resp.content)
            else:
                raise ValueError(
                    f"Failed to download file from {url}. HTTP Status Code: {resp.status_code}. Message: {resp.content}"
                )

        if self == CALMDownloadSite.BARROW:
            url = urljoin(
                DATA_PARENT_FOLDER, "North%20America/Alaska/North%20Slope/u01_barrow_grid/U1_alt_1995_2022.xls"
            )
            savepath = CALM_RAW_DATA_DIR / "U1_alt_1995_2022.xls"
            download_file(url, savepath)
            # We may need to use other sheets in the future
            df = pd.read_excel(savepath, sheet_name="data")
            df = df[:121]  # 121 nodes. Ignore the last few points cause they are averages. We don't need them.
            df = df.rename(
                {
                    "GridNode": StandardDataFormatColumns.POINT_ID.value,
                    "Latitude": StandardDataFormatColumns.LATITUDE.value,
                    "Longitude": StandardDataFormatColumns.LONGITUDE.value,
                },
                axis=1,
            )
            alt_cols = [col for col in df.columns if col.startswith("al-")]
            used_cols = [
                StandardDataFormatColumns.POINT_ID.value,
                StandardDataFormatColumns.LATITUDE.value,
                StandardDataFormatColumns.LONGITUDE.value,
            ]

            def parse_date(alt_col):
                # examples: al-080295, al-81409
                numeric_part = alt_col.split("-")[-1]

                try:
                    return datetime.datetime.strptime(numeric_part, "%m%d%y")
                except Exception as e:
                    raise ValueError(f"Failed to parse col {alt_col}. Error: {e}")

            date_dfs = []
            for date_col in alt_cols:
                df_date = df[used_cols + [date_col]].copy()
                # Create a new column specifying the date
                df_date[StandardDataFormatColumns.DATE.value] = parse_date(date_col)
                df_date = df_date.rename({date_col: StandardDataFormatColumns.ALT_METERS.value}, axis=1)
                date_dfs.append(df_date)

            df_all = pd.concat(date_dfs, axis=0, ignore_index=True, verify_integrity=True)
            StandardDataFormatColumns.save_standardized_dataframe(df_all, "u1")

            print("TODO: Download U2 data")
        else:
            raise ValueError(f"Could not match {self}?")


@click.command()
@click.option(
    "--site",
    type=click.Choice([e.value for e in CALMDownloadSite], case_sensitive=False),
    help="CALM Site from which to download.",
)
def calm(site: Optional[str]):
    sites = None
    if site is not None:
        sites = [site]
    else:
        print("Downloading data for all CALM sites...")
        sites = [site for site in CALMDownloadSite]

    pbar = tqdm.tqdm(sites)
    for site in pbar:
        # adds typing annotation
        site: CALMDownloadSite = site
        pbar.set_description(f"Downloading {site.value}")
        site.download()


@click.group()
def main():
    pass


main.add_command(calm)
main.add_command(alos_palsar_granule)

if __name__ == "__main__":
    main()
