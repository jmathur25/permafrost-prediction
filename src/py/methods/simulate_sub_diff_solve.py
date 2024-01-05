from matplotlib import pyplot as plt
import numpy as np

from methods.soil_models import ConstantWaterSMM, SoilMoistureModel
from scipy.stats import pearsonr
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar

def generate_h1_h2_subs(sqrt_ddt_ref, sqrt_ddt_sec, smm: SoilMoistureModel, upper_thaw_depth_limit=1.0, N=1000):
    Q = sqrt_ddt_ref/sqrt_ddt_sec
    # assert sqrt_ddt_ratio > 1.0
    if Q > 1.0:
        # Under this condition, h1 < h2 so to make h2 upper bounded by `upper_thaw_depth_limit`,
        # we need h1 upper bounded by `upper_thaw_depth_limit/Q`
        # Under the opposite condition, h1 > h2 so `upper_thaw_depth_limit` is correct as it is.
        upper_thaw_depth_limit = upper_thaw_depth_limit/Q
    h1s = np.linspace(0.01, upper_thaw_depth_limit, num=N)
    h2s = Q * h1s
    subsidence_differences = []
    for h2, h1 in zip(h2s, h1s):
        sub = smm.deformation_from_alt(h2) - smm.deformation_from_alt(h1)
        subsidence_differences.append(sub)
    subsidence_differences = np.array(subsidence_differences)
    return h1s, h2s, subsidence_differences


# TODO: what to do if thaw has not started?
def find_best_alt_diff(deformation_per_pixel, sqrt_ddt_ref, sqrt_ddt_sec, smm: SoilMoistureModel, upper_thaw_depth_limit=1.0):
    # TODO: precompute?
    h1s, h2s, subsidence_differences = generate_h1_h2_subs(sqrt_ddt_ref, sqrt_ddt_sec, smm)
    sorter = np.argsort(subsidence_differences)
    thaw_depth_differences = []
    for d in deformation_per_pixel:
        idx = np.searchsorted(subsidence_differences, d, sorter=sorter)
        if idx == len(subsidence_differences):
            idx = len(subsidence_differences) - 1
        thaw_depth_differences.append(h2s[idx] - h1s[idx])
    return thaw_depth_differences


if __name__ == '__main__':
    ddt_ref = 225.0
    ddt_sec = 10.0
    sqrt_ddt_ref = np.sqrt(ddt_ref)
    sqrt_ddt_sec = np.sqrt(ddt_sec)
    smm = ConstantWaterSMM(s=0.5)
    ret = find_best_alt_diff([0.01], TODO)
    print(ret)
