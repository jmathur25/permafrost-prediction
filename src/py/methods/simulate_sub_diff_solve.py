from matplotlib import pyplot as plt
import numpy as np

from methods.soil_models import ConstantWaterSMM, SoilMoistureModel
from scipy.stats import pearsonr
from scipy.optimize import fsolve


def generate_h1_h2_subs(sqrt_ddt_ref, sqrt_ddt_sec, smm: SoilMoistureModel, upper_thaw_depth_limit=1.0, N=1000):
    Q = sqrt_ddt_ref/sqrt_ddt_sec
    # assert sqrt_ddt_ratio > 1.0
    if Q > 1.0:
        # Under this condition, h1 < h2 so to make h2 upper bounded by `upper_thaw_depth_limit`,
        # we need h1 upper bounded by `upper_thaw_depth_limit/Q`
        # Under the opposite condition, h1 > h2 so `upper_thaw_depth_limit` is correct as it is.
        upper_thaw_depth_limit = upper_thaw_depth_limit/Q
    h1s = np.linspace(0.001, upper_thaw_depth_limit, num=N)
    h2s = Q * h1s
    subsidences = []
    for h2, h1 in zip(h2s, h1s):
        sub = smm.deformation_from_alt(h2) - smm.deformation_from_alt(h1)
        subsidences.append(sub)
    subsidences = np.array(subsidences)
    return h1s, h2s, subsidences


# TODO: what to do if thaw has not started?
def find_best_alt_diff(deformation_per_pixel, sqrt_ddt_ref, sqrt_ddt_sec, smm: SoilMoistureModel):
    # print(sqrt_ddt_ref, sqrt_ddt_sec)
    # min_v, max_v = sqrt_ddt_sec, sqrt_ddt_ref
    # if min_v > max_v:
    #     max_v, min_v = min_v, max_v
    # thresh = 0.2
    # if min_v < thresh:
    #     print("Invoking thresh")
    #     return [smm.alt_from_deformation(d) if d > 1e-3 else 0 for d in deformation_per_pixel]
    h1s, h2s, subsidences = generate_h1_h2_subs(sqrt_ddt_ref, sqrt_ddt_sec, smm)
    # TODO: find min...
    sorter = np.argsort(subsidences)
    alt_diff = []
    for d in deformation_per_pixel:
        idx = np.searchsorted(subsidences, d, sorter=sorter)
        if idx == len(subsidences):
            # if subsidence is too large, round to nearest? TODO: warning?
            idx = len(subsidences) - 1
        alt_diff.append(h2s[idx] - h1s[idx])
    return alt_diff


if __name__ == '__main__':
    ddt_ref = 225.0
    ddt_sec = 10.0
    sqrt_ddt_ref = np.sqrt(ddt_ref)
    sqrt_ddt_sec = np.sqrt(ddt_sec)
    smm = ConstantWaterSMM(s=0.5)
    ret = find_best_alt_diff([0.01], sqrt_ddt_ratio)
    print(ret)
