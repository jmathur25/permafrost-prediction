"""
This is used to save space if interferograms are taking up too much space on your machine.
"""

import os
import shutil

from pp.methods.igrams import SCHAEFER_INTEFEROGRAMS
from pp.data.consts import ISCE2_OUTPUTS_DIR

useless = [
    "coregisteredSlc/highBand",
    "coregisteredSlc/lowBand",
    "interferogram/highBand",
    "interferogram/lowBand",
    "offsets",
    "coregisteredSlc"
]

for (alos1, alos2) in SCHAEFER_INTEFEROGRAMS:
    for u in useless:
        p = ISCE2_OUTPUTS_DIR / f'{alos1}_{alos2}' / u
        if (p.exists()):
            print("deleting", p)
            shutil.rmtree(p)
