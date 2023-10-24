# %%
from osgeo import gdal
# %%

# ds = gdal.Open('/permafrost-prediction-shared-data/isce2_outputs/ALPSRP021272170_ALPSRP027982170/ALPSRP021272170_slc/ALPSRP021272170.slc', gdal.GA_ReadOnly)
# assert ds.RasterCount == 1

ds2 = gdal.Open('/permafrost-prediction-shared-data/isce2_outputs/ALPSRP021272170_ALPSRP027982170/ALPSRP027982170_slc/ALPSRP027982170.slc', gdal.GA_ReadOnly)
assert ds2.RasterCount == 1

# %%
# data = ds.GetRasterBand(1).ReadAsArray()
data2 = ds2.GetRasterBand(1).ReadAsArray()

# %%
