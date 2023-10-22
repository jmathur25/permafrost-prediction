from abc import ABC, abstractmethod
from enum import Enum
import pathlib
from haversine import haversine, Unit
from isce.components import isceobj
import numpy as np
from osgeo import gdal
import pandas as pd
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

    def find_closest_pixel(self, lat, lon, max_dist_meters=35):
        _, closest_pixel_idx_flattened = self.kd_tree.query([lat, lon])
        closest_pixel_idx = np.unravel_index(closest_pixel_idx_flattened, self.lat_arr.shape)
        # TODO: implement proper distance checking
        dist = haversine((lat, lon), (self.lat_arr[closest_pixel_idx], self.lon_arr[closest_pixel_idx]), unit=Unit.METERS)
        assert dist < max_dist_meters
        return closest_pixel_idx


def compute_stats(alt_pred, alt_gt):
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
    
def load_calm_data(calm_file, ignore_point_ids, start_year, end_year):
    df_calm = pd.read_csv(calm_file, parse_dates=["date"])
    df_calm = df_calm.sort_values("date", ascending=True)
    df_calm = df_calm[df_calm["point_id"].apply(lambda x: x not in ignore_point_ids)]
    # TODO: can we just do 2006-2010? Before 1995-2013
    # pandas version 2.1.1 has a bug where <= is not-inclusive. So we do: < pd.to_datetime(str(end_year + 1)
    df_calm = df_calm[(df_calm["date"] >= pd.to_datetime(str(start_year))) & (df_calm["date"] < pd.to_datetime(str(end_year + 1)))]

    def try_float(x):
        try:
            return float(x)
        except:
            return np.nan

    # TODO: fix in processor. handle 'w'?
    df_calm["alt_m"] = df_calm["alt_m"].apply(try_float) / 100
    # only grab ALTs from end of summer, which will be the last
    # measurement in a year
    df_calm["year"] = df_calm["date"].dt.year
    df_peak_alt = df_calm.groupby(["point_id", "year"]).last().reset_index()
    df_peak_alt['month'] = df_peak_alt['date'].dt.month
    df_peak_alt['day'] = df_peak_alt['date'].dt.day
    return df_peak_alt 

def prepare_calm_data(calm_file, ignore_point_ids, start_year, end_year, ddt_scale, df_temp):
    df_peak_alt = load_calm_data(calm_file, ignore_point_ids, start_year, end_year)
    df_peak_alt = pd.merge(df_peak_alt, df_temp[['ddt', 'norm_ddt']], on=['year', 'month', 'day'], how='left')
    df_max_yearly_ddt = df_temp.groupby('year').last()[['norm_ddt']]
    df_max_yearly_ddt = df_max_yearly_ddt.rename({'norm_ddt': 'max_yearly_ddt'}, axis=1)
    df_peak_alt = pd.merge(df_peak_alt, df_max_yearly_ddt, on='year', how='left')
    if ddt_scale:
        df_peak_alt['alt_m'] = df_peak_alt['alt_m'] * (df_peak_alt['max_yearly_ddt'] / df_peak_alt['norm_ddt'])
    return df_peak_alt.drop(['date', 'max_yearly_ddt'], axis=1)

def prepare_temp(temp_file, start_year, end_year, norm_per_year):
    df_temp = pd.read_csv(temp_file)
    assert len(pd.unique(df_temp["site_code"])) == 1  # TODO: support codes
    df_temp = df_temp[(df_temp["year"] >= start_year) & (df_temp["year"] <= end_year)]
    df_temp = df_temp.sort_values(["year", "month", "day"]).set_index(["year", "month", "day"])
    df_temps = []
    for year, df_t in df_temp.groupby("year"):
        compute_ddt_ddf(df_t)
        if norm_per_year:
            # QUESTION: is "end of season" normalization when barrow measures or just highest DDT per year?
            df_t["norm_ddt"] = df_t["ddt"] / df_t["ddt"].values[-1]
            # date_max_alt = df_peak_alt.loc[calib_point_id, year]["date"]
            # end_of_season_ddt = df_t.loc[year, date_max_alt.month, date_max_alt.day]["ddt"]
            # df_t["norm_ddt"] = df_t["ddt"] / end_of_season_ddt
        df_temps.append(df_t)
    df_temp = pd.concat(df_temps, verify_integrity=True)
    if not norm_per_year:
        max_ddt = df_temp["ddt"].max()
        # print("overring max ddt")
        # max_ddt = 15**2
        df_temp["norm_ddt"] = df_temp["ddt"] / max_ddt
        # df_temp[df_temp['norm_ddt'] > 1.0] = 1.0
    return df_temp

def compute_ddt_ddf(df):
    tmp_col = "temp_2m_c"
    freezing_point = 0.0
    # Initialize counters
    ddf = 0
    ddt = 0
    ddf_list = []
    ddt_list = []

    # Iterate over the temp_2m_c column
    for i, temp in enumerate(df[tmp_col]):
        if not np.isnan(temp):
            if temp < freezing_point:
                ddf += temp
            else:
                ddt += temp
        ddf_list.append(ddf)
        ddt_list.append(ddt)

    df["ddf"] = ddf_list
    df["ddt"] = ddt_list
