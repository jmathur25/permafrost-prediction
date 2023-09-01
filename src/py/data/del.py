from consts import ALOS_PALSAR_DATA_DIR
import os
import shutil 
ok_list = [
    "ALPSRP074952170",
    "ALPSRP021272170",
    "ALPSRP027982170"
]

for f in os.listdir(ALOS_PALSAR_DATA_DIR):
    if f not in ok_list:
        shutil.rmtree(ALOS_PALSAR_DATA_DIR / f)
