from abc import ABC, abstractmethod
from enum import Enum
import pathlib
from haversine import haversine
from isce.components import isceobj
import numpy as np
from osgeo import gdal
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
import h5py

# from DownsampleUnwrapper
def load_img(xml_path):
    assert xml_path.suffixes == [".slc", ".xml"]
    img = isceobj.createImage()
    img.load(xml_path)
    _dtype_map = {"cfloat": np.complex64, "float": np.float32, "byte": np.uint8}
    dtype = _dtype_map[img.dataType.lower()]
    width = img.getWidth()
    length = img.getLength()
    im = np.fromfile(str(xml_path)[:-4], dtype)
    assert img.bands in [1, 2], f"Unsupported number of bands: {img.bands}"
    if img.bands == 1:
        im = im.reshape([length, width])
    else:  # the other option is the unw which is 2 bands BIL
        im = im.reshape([length, img.bands, width])
    return im


class LatLonFile(Enum):
    RDR = "rdr" # ISCE2 output in radar image frame
    H5 = "h5" # MintPy output after geocoding
    XML = "xml" # ISCE2 output after geocoding
    
    
    def create_lat_lon(self, path: pathlib.Path) -> "LatLon":
        if self == LatLonFile.RDR:
            ds = gdal.Open(str(path / "lat.rdr"), gdal.GA_ReadOnly)
            lat = ds.GetRasterBand(1).ReadAsArray()
            ds = None

            # reading the lat/lon
            ds = gdal.Open(str(path / "lon.rdr"), gdal.GA_ReadOnly)
            lon = ds.GetRasterBand(1).ReadAsArray()
            del ds
            return LatLonArray(lat, lon)
        elif self == LatLonFile.H5:
            geo_file = path / "geo_geometryRadar.h5"
            geo_data = h5py.File(geo_file)
            lat = geo_data['latitude'][()]
            lon = geo_data['longitude'][()]
            return LatLonArray(lat, lon)
        elif self == LatLonFile.XML:
            return LatLonXML(path)

class LatLon(ABC):
    
    @abstractmethod
    def find_closest_pixel(self, lat, lon):
        pass
    
    
class LatLonXML(LatLon):
    def __init__(self, intfg: pathlib.Path):
        assert intfg.suffix == '.geo'
        self.root = ET.parse(str(intfg) + ".xml")
        self.start_lon, self.delta_lon = self.extract_values("coordinate1")
        self.start_lat, self.delta_lat = self.extract_values("coordinate2")
        vert_res = (
            haversine((self.start_lat, self.start_lon), (self.start_lat + 1000 * self.delta_lat, self.start_lon)) / 1000
        ) * 1000
        horiz_res = (
            haversine((self.start_lat, self.start_lon), (self.start_lat, 1000 * self.delta_lon + self.start_lon)) / 1000
        ) * 1000
        print("VERT RES (m):", vert_res)
        print("HORIZ RES (m):", horiz_res)

    def extract_values(self, coordinate_name):
        coord = self.root.find(f".//component[@name='{coordinate_name}']")
        starting_value = coord.find(".//property[@name='startingvalue']/value").text
        delta = coord.find(".//property[@name='delta']/value").text
        return float(starting_value), float(delta)

    def find_closest_pixel(self, lat, lon):
        # int() led to perfect alignment with another tool that converted the .geo
        # into a .tif, and then using a lat/lon to pixel function on the .tif.
        # round() led to discrepencies.
        y = int((lat - self.start_lat) / self.delta_lat)
        x = int((lon - self.start_lon) / self.delta_lon)
        assert y >= 0
        assert x >= 0
        return y, x
    


class LatLonArray(LatLon):
    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat_arr = lat
        self.lon_arr = lon
        
        # lat/lon can be nan is doing geocoding, as the radar image
        # will not be perfectly rectangular in Earth lat/lon coordinate system
        not_nans = np.argwhere(~np.isnan(self.lat_arr))
        idx1 = not_nans[0]
        idx2 = not_nans[-1]
        dy_pix = idx2[0] - idx1[0]
        dx_pix = idx2[1] - idx1[1]
        # Cannot make meaningful claims on pixel resolution unless we have enough
        assert dy_pix > 100
        assert dx_pix > 100
        diag_dist_meters = 1000 * haversine((self.lat_arr[idx2[0], idx2[1]], self.lon_arr[idx2[0], idx2[1]]), (self.lat_arr[idx1[0], idx1[1]], self.lon_arr[idx1[0], idx1[1]]))
        theta = np.arctan2(dy_pix, dx_pix)
        dy_meters = diag_dist_meters * np.sin(theta)
        dx_meters = diag_dist_meters * np.cos(theta)

        print("VERT RES (m)", dy_meters / dy_pix)
        print("HORIZ RES (m)", dx_meters / dx_pix)

        self.flattened_coordinates = np.column_stack((self.lat_arr.ravel(), self.lon_arr.ravel()))
        self.kd_tree = KDTree(self.flattened_coordinates)

    def find_closest_pixel(self, lat, lon):
        _, closest_pixel_idx_flattened = self.kd_tree.query([lat, lon])
        closest_pixel_idx = np.unravel_index(closest_pixel_idx_flattened, self.lat_arr.shape)
        return closest_pixel_idx


def compute_stats(alt_pred, alt_gt, points):
    nan_mask_1 = np.isnan(alt_pred)
    nan_mask_2 = np.isnan(alt_gt)
    print(f"number of nans PRED: {nan_mask_1.sum()}/{len(alt_pred)}")
    print(f"number of nans GT: {nan_mask_2.sum()}/{len(alt_pred)}")
    not_nan_mask = ~(nan_mask_1 | nan_mask_2)
    alt_pred = alt_pred[not_nan_mask]
    alt_gt = alt_gt[not_nan_mask]
    diff = alt_pred - alt_gt
    print("ABS DIFF MEAN", np.mean(np.abs(diff)))
    e = 0.079
    # computing proper uncertainty involves propogating model parameter error down
    # (see https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JF002041)
    # paper mentions that model uncertainty is ~2x measurement uncertainty, hence this appx
    resalt_e = 2 * e
    chi_stat = np.square(diff / e)
    mask_is_great = chi_stat < 1
    alt_within_uncertainty_mask = (alt_pred - resalt_e < alt_gt) & (alt_gt < alt_pred + resalt_e)
    alt_within_uncertainty_mask &= ~mask_is_great  # exclude ones that are great
    print("FOR ENTIRE INPUT (excluding nans):")
    _print_stats(alt_pred, alt_gt, diff, chi_stat, mask_is_great, alt_within_uncertainty_mask)

    indices = np.argsort(chi_stat)
    print(f"\nFOR INPUT EXCLUDING THE WORST 3 POINTS: {points[indices[-3:]]}")
    _print_stats(
        alt_pred[indices][:-3],
        alt_gt[indices][:-3],
        diff[indices][:-3],
        chi_stat[indices][:-3],
        mask_is_great[indices][:-3],
        alt_within_uncertainty_mask[indices][:-3],
    )


def _print_stats(alt_pred, alt_gt, diff, chi_stat, mask_is_great, alt_within_uncertainty_mask):
    print("avg, std ALT pred", np.mean(alt_pred), np.std(alt_pred))
    print("avg, std ALT GT", np.mean(alt_gt), np.std(alt_gt))

    chi_stat_mean = np.mean(chi_stat)
    print("chi^2 avg", chi_stat_mean)
    frac_great_match = mask_is_great.mean()

    frac_good_match = alt_within_uncertainty_mask.mean()

    frac_bad_match = 1.0 - frac_great_match - frac_good_match

    print(f"frac great match: {frac_great_match}, frac good match: {frac_good_match}, frac bad match: {frac_bad_match}")

    bias = diff.mean()
    print("Bias", bias)

    r2 = r2_score(alt_pred, alt_gt)
    print(f"R^2 score: {r2}")

    pearson_corr, _ = pearsonr(alt_pred, alt_gt)
    print(f"Pearson R: {pearson_corr}")

    rmse = np.sqrt(mean_squared_error(alt_pred, alt_gt))
    print(f"RMSE: {rmse}")
