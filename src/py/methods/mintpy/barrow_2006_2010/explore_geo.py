# %%
%load_ext autoreload
%autoreload 2

# %%
import h5py

# %%
geo_ts = h5py.File("geo/geo_timeseries_tropHgt_demErr.h5")
date = geo_ts['date'][()]
timeseries = geo_ts['timeseries'][()]

# %%
geo_radar = h5py.File("geo/geo_geometryRadar.h5")
lat = geo_radar['latitude'][()]
lon = geo_radar['longitude'][()]
# %%
