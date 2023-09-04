

import pathlib

_CURRENT_DIR = pathlib.Path(__file__).parent
CALM_DOWNLOAD_URL = "https://www2.gwu.edu/~calm/data/CALM_Data/"

TEMP_DATA_DIR = _CURRENT_DIR / "temperature"

CALM_DATA_DIR = _CURRENT_DIR / "calm"
CALM_RAW_DATA_DIR = CALM_DATA_DIR / "raw_data"
CALM_PROCESSSED_DATA_DIR = CALM_DATA_DIR / "data"

ALOS_PALSAR_DATA_DIR = _CURRENT_DIR / "alos_palsar"
ALOS_L1_0_DIRNAME = "l1.0_data"

ISCE2_OUTPUTS_DIR = _CURRENT_DIR / "isce2_outputs"
