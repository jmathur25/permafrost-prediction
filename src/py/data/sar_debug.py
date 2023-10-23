#%%
import asf_search as asf
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# %%
all_granules = ['ALPSRP021272170', 'ALPSRP017332170', 'ALPSRP018792170', 'ALPSRP016671410', 'ALPSRP026522170', 'ALPSRP020901410', 'ALPSRP019812170', 'ALPSRP017401410']
results = asf.granule_search(all_granules)

# %%
matches = [r for r in results if r.meta['native-id'].endswith('-L1.0')]
assert len(matches) == len(all_granules)

# %%
# Parse the coordinates and create a Polygon
polygons = [Polygon(m.geometry['coordinates'][0]) for m in matches]

# %%
point = Point(-156.589301, 71.310720)
for m, r in zip(polygons, matches):
    if not m.contains(point):
        print(f"Granule {r.meta['native-id']} failed")
    else:
        print(f"Granule {r.meta['native-id']} ok")

# %%
gdf = gpd.GeoDataFrame({'geometry': [polygon1, polygon2, point], 'label': [matches[0].meta['native-id'], matches[1].meta['native-id'], 'Node 61']})

ax = gdf.plot(edgecolor="k", facecolor="none", markersize=100)

# Label the polygons and point
for geometry, label in zip(gdf.geometry, gdf['label']):
    if isinstance(geometry, Point):
        x, y = geometry.x, geometry.y
    else:
        x, y = geometry.centroid.x, geometry.centroid.y
    ax.text(x, y, label, fontsize=9, ha='center')

plt.show()

# %%
