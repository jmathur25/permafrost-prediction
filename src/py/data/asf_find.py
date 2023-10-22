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
wkt = 'POLYGON((-157.501 70.8686,-155.355 70.8686,-155.355 71.5596,-157.501 71.5596,-157.501 70.8686))'
s = time.time()
results = asf.geo_search(platform=[asf.PLATFORM.ALOS], intersectsWith=wkt)
e = time.time()
print("Took:", e - s, "seconds")

# %%
expected_granules = []
for (g1, g2) in SCHAEFER_INTEFEROGRAMS:
    expected_granules.append(g1)
    expected_granules.append(g2)
expected_granules = [e + '-L1.5' for e in expected_granules]
expected_granules = set(expected_granules)

# %%
# SANITY CHECK
matches = []
for r in results:
    if r.meta['native-id'] in expected_granules:
        print("Found", r.meta['native-id'])
        matches.append(r)
assert len(matches) == len(expected_granules)

# %%
results_filtered = [r for r in results if r.meta['native-id'].endswith('-L1.5')]

# %%
year = 2006
months = [5, 6, 7, 8]
matches = dict()
for r in results_filtered:
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

# Node 61 U1 plot
point = Point(-156.589301, 71.310720)
download_list = []
for (k, v) in matches.items():
    min_dist = float('inf')
    best_r = None
    for r in v:
        polygon = Polygon(r.geometry['coordinates'][0])
        distance = point.distance(polygon.centroid)
        if distance < min_dist:
            distance = min_dist
            best_r = r
    assert best_r is not None
    # to ignore -L1.5
    granule = best_r.meta['native-id'].split('-')[0]
    download_list.append((k, granule))

for date, granule in download_list:
    print(f"For {date}, download {granule}")
    



# %%
