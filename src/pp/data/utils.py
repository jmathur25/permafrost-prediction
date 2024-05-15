import datetime
import os
from typing import Tuple

from pp.data.consts import ALOS_L1_0_DIRNAME, ALOS_PALSAR_DATA_DIR


def prompt_user(message) -> bool:
    response = input(f"{message} (y/n): ").strip().lower()
    return response == "y"


def get_date_for_alos(alos) -> Tuple[str, datetime.datetime]:
    searchdir = ALOS_PALSAR_DATA_DIR / alos / ALOS_L1_0_DIRNAME
    files = os.listdir(searchdir)
    # 2 expected, and also pruning can leave just 1.
    assert len(files) in [1, 2], f"Unexpected number of files in {searchdir}"

    if len(files) == 2:
        # the name that is not ARCHIVED_FILES is our date
        datetime_str = files[0] if files[1] == "ARCHIVED_FILES" else files[1]
    else:
        datetime_str = files[0]
    return datetime_str, datetime.datetime.strptime(datetime_str, "%Y%m%d")
