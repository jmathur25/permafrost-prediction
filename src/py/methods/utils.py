from haversine import haversine
from isce2.components import isceobj
import numpy as np
import gdal
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


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

        horiz_dist = haversine((lat[0, 0], lon[0, 0]), (lat[0, -1], lon[0, -1]))
        vert_dist = haversine((lat[0, 0], lon[0, 0]), (lat[-1, 0], lon[-1, 0]))
        print("HORIZ RES (m)", horiz_dist / lat.shape[1] * 1000)
        print("VERT RES (m)", vert_dist / lat.shape[0] * 1000)

    def find_closest_pixel(self, lat, lon):
        distances = np.sqrt((self.lat_arr - lat) ** 2 + (self.lon_arr - lon) ** 2)
        closest_pixel_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
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
    resalt_e = 2 * e
    psi_stat = np.square(diff / e)
    mask_is_great = psi_stat < 1
    alt_within_uncertainty_mask = (alt_pred - resalt_e < alt_gt) & (alt_gt < alt_pred + resalt_e)
    alt_within_uncertainty_mask &= ~mask_is_great  # exclude ones that are great
    print("FOR ENTIRE INPUT (excluding nans):")
    _print_stats(alt_pred, alt_gt, diff, psi_stat, mask_is_great, alt_within_uncertainty_mask)

    print("\nFOR INPUT EXCLUDING THE WORST 3 POINTS:")
    indices = np.argsort(psi_stat)
    _print_stats(
        alt_pred[indices][:-3],
        alt_gt[indices][:-3],
        diff[indices][:-3],
        psi_stat[indices][:-3],
        mask_is_great[indices][:-3],
        alt_within_uncertainty_mask[indices][:-3],
    )


def _print_stats(alt_pred, alt_gt, diff, psi_stat, mask_is_great, alt_within_uncertainty_mask):
    print("avg, std ALT pred", np.mean(alt_pred), np.std(alt_pred))
    print("avg, std ALT GT", np.mean(alt_gt), np.std(alt_gt))

    psi_stat_mean = np.mean(psi_stat)
    print("psi avg", psi_stat_mean)
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
