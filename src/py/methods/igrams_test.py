
import sys

sys.path.append("/permafrost-prediction/src/py")
from methods.igrams import SCHAEFER_INTEFEROGRAMS
from data.utils import get_date_for_alos

for (alos2, alos1) in SCHAEFER_INTEFEROGRAMS:
    d2 = get_date_for_alos(alos2)
    d1 = get_date_for_alos(alos1)
    
    print("Alos d2,d1:", d2, d1)
