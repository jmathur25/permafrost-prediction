
import sys

sys.path.append("/permafrost-prediction/src/py")
from methods.igrams import JATIN_SINGLE_SEASON_2006_IGRAMS
from data.utils import get_date_for_alos

for (alos2, alos1) in JATIN_SINGLE_SEASON_2006_IGRAMS:
    d2 = get_date_for_alos(alos2)
    d1 = get_date_for_alos(alos1)
    
    print("Alos d2,d1:", d2, d1)
