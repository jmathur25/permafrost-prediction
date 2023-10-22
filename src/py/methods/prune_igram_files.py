import pathlib
import shutil

from igrams import JATIN_SINGLE_SEASON_2006_IGRAMS, SCHAEFER_INTEFEROGRAMS

useless = [
    "coregisteredSlc/highBand",
    "coregisteredSlc/lowBand",
    "interferogram/highBand",
    "interferogram/lowBand",
    "offsets",
    "coregisteredSlc"
]

for (alos1, alos2) in SCHAEFER_INTEFEROGRAMS:
    if (alos1, alos2) not in JATIN_SINGLE_SEASON_2006_IGRAMS:
        for u in useless:
            p = pathlib.Path(f"/permafrost-prediction-shared-data/isce2_outputs/{alos1}_{alos2}/{u}")
            if (p.exists()):
                print("deleting", p)
                shutil.rmtree(p)
