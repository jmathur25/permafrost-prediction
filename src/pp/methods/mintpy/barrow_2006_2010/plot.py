"""
Not relevant to paper. TODO: document/clean up
"""

# %%
%load_ext autoreload
%autoreload 2

# %%
import pathlib
from mintpy.cli import view, tsview, plot_network, plot_transection
import h5py
import sys
sys.path.append("../../../../")
from pp.methods.utils import LatLon
import matplotlib.pyplot as plt
from osgeo import gdal


# %%
plot_network.main('inputs/ifgramStack.h5 -t smallbaselineApp.cfg --figsize 12 4'.split())

# %%
view.main('avgSpatialCoh.h5 --noverbose'.split())

# %%
view.main('temporalCoherence.h5 --noverbose'.split())

# %%
ref = pathlib.Path("../Igrams/20060618_20060803/filt_20060618_20060803.cor")
ll = LatLon(
    pathlib.Path("../geom_reference"),
    # ref
    )

# %%
ll.find_closest_pixel(71.31072, -156.5893005)
# %%
%matplotlib widget
tsview.main('timeseries.h5 --ref-date 20060618 --ylim -8 8 --yx 1456 995 --noverbose --nodisplay'.split())

# %%
f = h5py.File('timeseries.h5', 'r') # timeseries.h5
los_def = f['timeseries'][()]
dates = f['date'][()]

# %%
print(los_def[:,2912,1327])
# %%
%matplotlib inline
plt.figure(figsize=(10, 6))
plt.plot(dates, los_def[:,2912,1327], marker='o')
plt.title('Timeseries Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
ds = gdal.Open("/permafrost-prediction/src/pp/data/alos_palsar/try_stack_stripmap/geom_reference/waterMask.rdr", gdal.GA_ReadOnly)
print(ds.RasterXSize, ds.RasterYSize)


# %%
ds = gdal.Open("/permafrost-prediction/src/pp/methods/swbdLat_N70_N72_Lon_W158_W154.wbd", gdal.GA_ReadOnly)
print(ds.RasterXSize, ds.RasterYSize)


# %%
f = h5py.File('geo/geo_timeseries_tropHgt_demErr.h5', 'r')
timeseries = f['timeseries'][()]
date = f['date'][()]
# %%
f = h5py.File('geo/geo_geometryRadar.h5', 'r')
lat = f['latitude'][()]
lon = f['longitude'][()]
# %%
