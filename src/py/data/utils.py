import datetime
import os

from data.consts import ALOS_L1_0_DIRNAME, ALOS_PALSAR_DATA_DIR


def prompt_user(message) -> bool:
    response = input(f"{message} (y/n): ").strip().lower()
    return response == "y"


def get_date_for_alos(alos) -> datetime.datetime:
    searchdir = ALOS_PALSAR_DATA_DIR / alos / ALOS_L1_0_DIRNAME
    files = os.listdir(searchdir)
    assert len(files) == 2, f"Unexpected number of files in {searchdir}"
    
    # the name that is not ARCHIVED_FILES is our date
    datetime_str = files[0] if files[1] == "ARCHIVED_FILES" else files[1]
    return datetime.datetime.strptime(datetime_str, "%Y%m%d")
    
