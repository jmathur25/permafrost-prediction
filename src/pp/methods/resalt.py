"""
Contains the ReSALT implementations. There are two varieties:
1. LIU_SCHAEFER. This is the existing ReSALT used in literature.
2. SCReSALT. This is the new method.
"""

from datetime import datetime
import enum
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm
from scipy.integrate import cumulative_trapezoid

from pp.methods.soil_models import SoilMoistureModel
from pp.methods.utils import get_norm_ddt

class ReSALT_Type(enum.Enum):
    LIU_SCHAEFER = 0
    SCReSALT = 1


class ReSALT:
    def __init__(self, df_temp: pd.DataFrame, smm: SoilMoistureModel, calib_idx: int, calib_deformation: float, calib_ddt: float, rtype: ReSALT_Type):
        """
        TODO: add support for gravel calibration point?
        """
        self.df_temp = df_temp
        self.calib_idx = calib_idx
        self.calib_deformation = calib_deformation
        self.calib_ddt = calib_ddt
        self.calib_alt = smm.alt_from_deformation(self.calib_deformation) if self.calib_deformation else None
        self.smm = smm
        self.rtype = rtype
     

    def run_inversion(self, deformations: np.ndarray, dates: List[Tuple[datetime, datetime]], only_solve_E: bool=False) -> np.ndarray:
        """
        deformations: 2D-array of (igram_idx, pixel) that stores deformation
        dates: list of dates of the igrams. It is assumed that the first index
            corresponds to the 'reference' and the second index to the 'secondary'. This means the deformation should be derived from:
            phase in the reference - phase in secondary.
        """
        n = deformations.shape[0]
        lhs_all = np.zeros((n, deformations.shape[1]))
        rhs_all = np.zeros((n, 2))
        
        calib_integral = self.smm.height_porosity_integration(0.0, self.calib_alt)
        calib_coeff = calib_integral / self.calib_ddt
        
        # TODO: integrate
        h1s = np.linspace(0.0, 1.0, num=1000)
        h1_integrands = []
        for h1 in h1s:
            h1_integrands.append(self.smm.height_porosity_integrand(h1))
        all_h1_int = cumulative_trapezoid(h1_integrands, h1s, initial=0)
        
        for i, (deformation_per_pixel, (date_ref, date_sec)) in tqdm.tqdm(enumerate(zip(deformations, dates)), total=len(deformations), desc="Processing each interferogram"):
            delta_t_years = (date_ref - date_sec).days / 365
            norm_ddt_ref = get_norm_ddt(self.df_temp, date_ref)
            norm_ddt_sec = get_norm_ddt(self.df_temp, date_sec)
            # sqrt_norm_ddt_ref = np.sqrt(norm_ddt_ref)
            # sqrt_norm_ddt_sec = np.sqrt(norm_ddt_sec)
            # sqrt_addt_diff = sqrt_norm_ddt_ref - sqrt_norm_ddt_sec
            rhs = [delta_t_years, norm_ddt_ref - norm_ddt_sec]
            
            if self.rtype == ReSALT_Type.LIU_SCHAEFER:
                # Solve as deltas over calibration deformation
                if self.calib_idx is not None:
                    delta = deformation_per_pixel[self.calib_idx]
                    lhs = deformation_per_pixel - delta
                else:
                    lhs = deformation_per_pixel
            elif self.rtype == ReSALT_Type.SCReSALT:
                if self.calib_alt:
                    idx = np.searchsorted(all_h1_int, calib_integral * norm_ddt_ref/self.calib_ddt)
                    if idx == len(all_h1_int):
                        idx -= 1
                    calib_ref_thaw_depth = h1s[idx]
                    
                    idx = np.searchsorted(all_h1_int, calib_integral * norm_ddt_sec/self.calib_ddt)
                    if idx == len(all_h1_int):
                        idx -= 1
                    calib_sec_thaw_depth = h1s[idx]
                    
                    calib_ref_subsidence = self.smm.deformation_from_alt(calib_ref_thaw_depth)
                    calib_sec_subsidence = self.smm.deformation_from_alt(calib_sec_thaw_depth)
                    calib_def = calib_ref_subsidence - calib_sec_subsidence
                    
                    # Scale the deformations to align with the expected calibration deformation
                    calib_node_delta = calib_def - deformation_per_pixel[self.calib_idx]
                    deformation_per_pixel = deformation_per_pixel + calib_node_delta
                
                tdd = find_best_thaw_depth_difference(deformation_per_pixel, norm_ddt_ref, norm_ddt_sec, self.smm)
                # Sanity check
                expected_alt_diff = calib_ref_thaw_depth - calib_sec_thaw_depth
                diff = tdd[self.calib_idx][1] - tdd[self.calib_idx][0]
                err = abs(diff - expected_alt_diff)
                assert err < 5e-3 # TODO: lower...
                lhs = []
                for (h1, h2) in tdd:
                    # TODO: just store this and pass it up/around?
                    lhi = self.smm.height_porosity_integration(h1, h2)
                    lhs.append(lhi)
            else:
                raise ValueError("Unknown rtype:", self.rtype)
                
                
            lhs_all[i,:] = lhs
            rhs_all[i,:] = rhs
        
        print("Solving equations")
        # SCReSALT mode cannot solve for R in current implementation
        if only_solve_E or self.rtype == ReSALT_Type.SCReSALT:
            rhs_all = rhs_all[:, [1]]
        rhs_pi = np.linalg.pinv(rhs_all)
        sol = rhs_pi @ lhs_all
        
        assert np.isnan(sol).sum() == 0
        if self.rtype == ReSALT_Type.SCReSALT:
            # TODO: below can help correct for any NANs. It is currently not encountered,
            # so this is left commented out.
            # alt_pred = sol[0, :]
            # # TOOD: explain
            # nans = np.argwhere(np.isnan(alt_pred))
            # if len(nans) > 0:
            #     resolved = 0
            #     for i in nans[:,0]:
            #         lhs_i = lhs_all[:,i]
            #         nan_mask = np.isnan(lhs_i)
            #         if np.mean(nan_mask) > 0.8:
            #             # Too many nans so bail.
            #             continue
            #         not_nan_mask = ~nan_mask
            #         alt_i = np.linalg.pinv(rhs_all[not_nan_mask]) @ lhs_i[not_nan_mask]
            #         assert alt_i.shape == (1,)
            #         alt_pred[i] = alt_i[0]
            #         resolved += 1
            #     print(f"Resolved {resolved}/{len(nans)} pixels with nans in least-squares inversion")
            alt_pred = []
            for n in sol[0,:]:
                alt_hat = self.smm.thaw_depth_from_addt_n(self.calib_ddt, n)
                alt_pred.append(alt_hat)
            alt_pred = np.array(alt_pred)
        elif self.rtype == ReSALT_Type.LIU_SCHAEFER:
            E_idx = 0 if only_solve_E else 1
            E = sol[E_idx, :]
            if self.calib_idx is not None:
                delta_E = self.calib_deformation - E[self.calib_idx]
                E = E + delta_E
            alt_pred = []
            for e in E:
                if e < 1e-3:
                    alt_pred.append(np.nan)
                    continue
                alt = self.smm.alt_from_deformation(e)
                alt_pred.append(alt)
            alt_pred = np.array(alt_pred)
        else:
            raise ValueError()
        return alt_pred


def find_best_thaw_depth_difference(deformation_per_pixel, ddt_ref, ddt_sec, smm: SoilMoistureModel, upper_thaw_depth_limit=1.0):
    """
    Implements Algorithm 1 in the paper.
    """
    subsidence_differences, h1s, h2s = generate_thaw_subsidence_differences(ddt_ref, ddt_sec, smm, upper_thaw_depth_limit)
    sorter = np.argsort(subsidence_differences)
    best_matching_thaw_depth_differences = []
    for i, d in enumerate(deformation_per_pixel):
        idx = np.searchsorted(subsidence_differences, d, sorter=sorter)
        if idx == len(subsidence_differences):
            # This means the subsidence difference is too large.
            idx = len(subsidence_differences) - 1
        h1 = h1s[sorter[idx]]
        h2 = h2s[sorter[idx]]
        best_matching_thaw_depth_differences.append((h1, h2))
    return best_matching_thaw_depth_differences
    
    
def generate_thaw_subsidence_differences(ddf_ref, ddt_sec, smm: SoilMoistureModel, upper_thaw_depth_limit: float, N=1000):
    """
    Generates thaw depth pairs and computes corresponding subsidences. Returns thaw depth differences and corresponding subsidence differences.
    
    Args:
        upper_thaw_depth_limit: Limit the sampled thaw depths to be below this number.
        N: Number of thaw depth pairs to generate.
    """
    Q = ddf_ref/ddt_sec
    # assert sqrt_ddt_ratio != 1.0 # or something close...
    h1s = np.linspace(0.0, upper_thaw_depth_limit, num=N)
    h1_integrands = []
    for h1 in h1s:
        h1_integrands.append(smm.height_porosity_integrand(h1))
    all_h1_int = cumulative_trapezoid(h1_integrands, h1s, initial=0)
    h2s = []
    for h1_int in all_h1_int:
        h2_int = Q * h1_int
        idx = np.searchsorted(all_h1_int, h2_int)
        if idx == len(h1s):
            # The h2 will exceed the thaw depth limit.
            break
        h2 = h1s[idx]
        assert h2 <= upper_thaw_depth_limit
        h2s.append(h2)
    # h2s can be of smaller length than h1s to enforce `upper_thaw_depth_limit`
    h2s = np.array(h2s)
    h1s = h1s[:len(h2s)]

    subsidence_differences = []
    for h2, h1 in zip(h2s, h1s):
        sub = smm.deformation_from_alt(h2) - smm.deformation_from_alt(h1)
        subsidence_differences.append(sub)
    subsidence_differences = np.array(subsidence_differences)
    return subsidence_differences, h1s, h2s


    

