import pathlib

WORK_DIR = pathlib.Path("/permafrost-prediction/work")
CALM_DOWNLOAD_URL = "https://www2.gwu.edu/~calm/data/CALM_Data/"

TEMP_DATA_DIR = WORK_DIR / "temperature"

CALM_DATA_DIR = WORK_DIR / "calm"
CALM_RAW_DATA_DIR = CALM_DATA_DIR / "raw_data"
CALM_PROCESSSED_DATA_DIR = CALM_DATA_DIR / "data"

ALOS_PALSAR_DATA_DIR = WORK_DIR / "alos_palsar"
ALOS_L1_0_DIRNAME = "l1.0_data"

ISCE2_OUTPUTS_DIR = WORK_DIR / "isce2_outputs"
STACK_STRIPMAP_OUTPUTS_DIR = WORK_DIR / "stack_stripmap_outputs"
