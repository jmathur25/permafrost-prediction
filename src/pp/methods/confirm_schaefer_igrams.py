"""
Given that there was some errors in listing the interferograms in Schaefer et al. (2015),
this checks that the interferograms match the expected dates and there are no duplicates.
"""

from datetime import datetime
import sys

from pp.methods.igrams import SCHAEFER_INTEFEROGRAMS
from pp.data.utils import get_date_for_alos


all_dates = [
    [20060618,	20060803],
    [20060618,	20080623],
    [20060618,	20090626],
    [20060618,	20090811],
    [20060803,	20090626],
    [20070621,	20070806],
    [20070621,	20080623],
    [20070621,	20090626],
    [20070621,	20090811],
	[20070621,	20100629],
	[20070806,	20080623],
	[20070806,	20090626],
	[20070806,	20090811],
	# [20070806,	20100629], # TODO: reprocess
	[20070806,	20100814],
	[20080623,	20090626],
	[20080623,	20090811],
	[20090626,	20090811],
	[20090811,	20100629],
	[20100629,	20100814],
]

for (d1, d2), (alos1, alos2) in zip(all_dates, SCHAEFER_INTEFEROGRAMS):
    d1 = datetime.strptime(str(d1), "%Y%m%d")
    d2 = datetime.strptime(str(d2), "%Y%m%d")
    _, alos_d1 = get_date_for_alos(alos1)
    _, alos_d2 = get_date_for_alos(alos2)
    assert d1 == alos_d1
    assert d2 == alos_d2


for i in range(len(SCHAEFER_INTEFEROGRAMS)):
    for j in range(i + 1, len(SCHAEFER_INTEFEROGRAMS)):
        if SCHAEFER_INTEFEROGRAMS[i] == SCHAEFER_INTEFEROGRAMS[j]:
            raise ValueError(f"row {i} = row {j}")
assert len(SCHAEFER_INTEFEROGRAMS) == 18

print("All ok!")
