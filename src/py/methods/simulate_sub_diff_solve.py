from matplotlib import pyplot as plt
import numpy as np

from methods.soil_models import ConstantWaterSMM, SoilMoistureModel
from scipy.stats import pearsonr


# TODO: what to do if thaw has not started?
def find_best_alt_diff(deformation_per_pixel, sqrt_ddt_ref, sqrt_ddt_sec, smm: SoilMoistureModel):
    sqrt_ddt_ratio = sqrt_ddt_ref/sqrt_ddt_sec
    # assert sqrt_ddt_ratio > 1.0
    h1s = np.linspace(0.01, 1.0, num=10000)
    h2s = sqrt_ddt_ratio * h1s
    subsidences = []
    for h2, h1 in zip(h2s, h1s):
        sub = smm.deformation_from_alt(h2) - smm.deformation_from_alt(h1)
        subsidences.append(sub)
    subsidences = np.array(subsidences)
    
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
