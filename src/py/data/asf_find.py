"""
Helper to download data using asf_search. Not needed for the paper because
the granules were given by Schaefer et al. (2015), but this is helpful
for finding new granules.
"""

# %%
from datetime import datetime
import asf_search as asf
import time
from shapely.geometry import Polygon, Point

import sys
sys.path.append("..")
from methods.igrams import SCHAEFER_INTEFEROGRAMS
# %%
# use https://search.asf.alaska.edu/
wkt = 'POINT (-156.589301 71.31072)'
s = time.time()
results = asf.geo_search(platform=[asf.PLATFORM.ALOS], intersectsWith=wkt, start=datetime(2006, 1, 1))
e = time.time()
print("Took:", e - s, "seconds")

# %%
expected_granules = []
for (g1, g2) in SCHAEFER_INTEFEROGRAMS:
    expected_granules.append(g1)
    expected_granules.append(g2)
expected_granules = [e + '-L1.0' for e in expected_granules]
expected_granules = set(expected_granules)

# %%
# SANITY CHECK
matches = []
for r in results:
    if r.meta['native-id'] in expected_granules:
        matches.append(r)
assert len(matches) == len(expected_granules)

# %%
results_filtered = [r for r in results if r.meta['native-id'].endswith('-L1.0')]
polygons = [Polygon(m.geometry['coordinates'][0]) for m in results_filtered]
point = Point(-156.589301, 71.310720)

# %%
year = 2006
months = [5, 6, 7, 8]
matches = dict()
for r, p in zip(results_filtered, polygons):
    if not p.contains(point):
        # print(f"Failed {r.meta['native-id']}")
        continue
    start_date = r.umm['TemporalExtent']['RangeDateTime']['BeginningDateTime']
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    if (start_date.year == year and start_date.month in months):
        print(f"Found granule on {start_date}:", r.meta['native-id'])
        # TODO: aggregate based on time?
        idx = (start_date.year, start_date.month, start_date.day)
        if idx not in matches:
            matches[idx] = []
        matches[(start_date.year, start_date.month, start_date.day)].append(r)


# %%

# TODO: see if this is really needed?
download_list = []
for (k, v) in matches.items():
    assert len(v) == 1
    # min_dist = float('inf')
    # best_r = None
    # for r in v:
    #     polygon = Polygon(r.geometry['coordinates'][0])
    #     distance = point.distance(polygon.centroid)
    #     if distance < min_dist:
    #         distance = min_dist
    #         best_r = r
    # assert best_r is not None
    # to ignore -L1.0
    best_r = v[0]
    
    granule = best_r.meta['native-id'].split('-')[0]
    download_list.append((k, granule))

for date, granule in download_list:
    print(f"For {date}, download {granule}")
    



# %%
