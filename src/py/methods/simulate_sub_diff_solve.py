from matplotlib import pyplot as plt
import numpy as np

from methods.soil_models import alt_to_surface_deformation
from scipy.stats import pearsonr


# TODO: what to do if thaw has not started?
def find_best_alt_diff(deformation_per_pixel, sqrt_ddt_ratio):
    assert sqrt_ddt_ratio > 1.0
    h1s = np.linspace(0.01, 0.09, num=100)
    h2s = sqrt_ddt_ratio * h1s
    subsidences = []
    for h2, h1 in zip(h2s, h1s):
        sub = alt_to_surface_deformation(h2) - alt_to_surface_deformation(h1)
        subsidences.append(sub)
    subsidences = np.array(subsidences)
    
    sorter = np.argsort(subsidences)
    alt_diff = []
    for d in deformation_per_pixel:
        idx = np.searchsorted(subsidences, d, sorter=sorter)
        if idx == len(subsidences):
            # subsidence is too large
            idx = len(subsidences) - 1
        alt_diff.append(h2s[idx] - h1s[idx])
    return alt_diff

if __name__ == '__main__':
    sqrt_ddt_ratio = 3.074552276325944
    ret = find_best_alt_diff([0.007657], sqrt_ddt_ratio)
    print(ret)
