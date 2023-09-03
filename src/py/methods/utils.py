
from isce2.components import isceobj
import numpy as np
import gdal

# from DownsampleUnwrapper
def load_img(xml_path):
    assert xml_path.suffixes == ['.slc', '.xml']
    img = isceobj.createImage()
    img.load(xml_path)
    _dtype_map = {'cfloat': np.complex64,'float': np.float32,'byte': np.uint8}
    dtype = _dtype_map[img.dataType.lower()]
    width = img.getWidth()
    length = img.getLength()
    im = np.fromfile(str(xml_path)[:-4], dtype)
    assert img.bands in [1, 2], f"Unsupported number of bands: {img.bands}"
    if img.bands == 1:
        im = im.reshape([length, width])
    else: #the other option is the unw which is 2 bands BIL
        im = im.reshape([length, img.bands, width])
    return im


class LatLon:
    def __init__(self, results_dir):
        # reading the lat/lon
        ds = gdal.Open(str(results_dir / "geometry/lat.rdr"), gdal.GA_ReadOnly)
        lat = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        
        # reading the lat/lon
        ds = gdal.Open(str(results_dir / "geometry/lon.rdr"), gdal.GA_ReadOnly)
        lon = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        self.lat_arr = lat
        self.lon_arr = lon

    def find_closest_pixel(self, lat, lon):
        distances = np.sqrt((self.lat_arr - lat)**2 + (self.lon_arr - lon)**2)
        closest_pixel_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        return closest_pixel_idx

