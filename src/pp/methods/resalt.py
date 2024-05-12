"""
Contains the ReSALT implementations. There are two varieties:
1. LIU_SCHAEFER. This is the existing ReSALT used in literature.
2. SCReSALT. This is the new method.
"""

from datetime import datetime
import enum
import traceback
from typing import List, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import tqdm

from pp.methods.soil_models import SoilDepthIntegration, SoilDepthIntegrationReturn, SoilMoistureModel
from pp.methods.utils import get_norm_ddt


class ReSALT_Type(enum.Enum):
    LIU_SCHAEFER = 0
    SCReSALT = 1
    SCReSALT_NS = 2
    
    
class ReSALT:
    """
    Solves the ReSALT inversion problem of inferring thaw depth from subsidence and temperature data.
    """
    def __init__(self, df_temp: pd.DataFrame, smm: SoilMoistureModel, calib_idx: int, calib_deformation: float, calib_ddts: np.ndarray, rtype: ReSALT_Type):
        """
        TODO: add support for gravel calibration point?
        
        Args:
            calib_ddts: List of (normalized) DDTS for the calibration point for all years under study. This is used to help scale the calibration ALT. For SCReSALT, we need the average all the square root DDTs. For SCReSALT, we can use the average DDT.
            
        TODO: it is confusing to provide the average calibration ALT but the full list of DDTs. This interface should be reworked.
        """
        self.df_temp = df_temp
        self.calib_idx = calib_idx
        self.calib_deformation = calib_deformation
        self.calib_ddt = np.mean(calib_ddts)
        self.calib_root_ddt = np.mean(np.sqrt(calib_ddts))
        self.calib_alt = smm.thaw_depth_from_deformation(self.calib_deformation) if self.calib_deformation else None
        self.smm = smm
        self.rtype = rtype
        
        # TODO: make args
        self.upper_thaw_depth_limit = 1.0
        self.N = 1000
        
        self.sdi = None
        self.calib_integral = None
        if rtype == ReSALT_Type.SCReSALT_NS:
            self.sdi = SoilDepthIntegration(smm, self.upper_thaw_depth_limit, self.N)
            self.calib_integral = self.smm.height_porosity_integration(0.0, self.calib_alt)
     

    def run_inversion(self, deformations: np.ndarray, dates: List[Tuple[datetime, datetime]], only_solve_E: bool=False, multithreaded=True) -> np.ndarray:
        """
        deformations: 2D-array of (igram_idx, pixel) that stores deformation
        dates: list of dates of the igrams. It is assumed that the first index
            corresponds to the 'reference' and the second index to the 'secondary'. This means the deformation should be derived from:
            phase in the reference - phase in secondary.
        """
        if only_solve_E:
            assert self.rtype == ReSALT_Type.LIU_SCHAEFER
            
        n = deformations.shape[0]
        lhs_all = np.zeros((n, deformations.shape[1]))
        n_rhs = 2 if self.rtype == ReSALT_Type.LIU_SCHAEFER else 1
        rhs_all = np.zeros((n, n_rhs))
        
        if multithreaded:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                pbar = tqdm.tqdm(total=n, desc='Processing each interferogram')
                futures = []
                for i, (deformation_per_pixel, (date_ref, date_sec)) in enumerate(zip(deformations, dates)):
                    f = executor.submit(_process_deformations_wrapper, self, i, deformation_per_pixel, date_ref, date_sec)
                    futures.append(f)
                
                for f in concurrent.futures.as_completed(futures):
                    result = f.result()
                    if isinstance(result, tuple) and isinstance(result[0], Exception):
                        print("Stacktrace:", result[1])
                        raise result[0]
                    else:
                        i, rhs, lhs = result
                        lhs_all[i,:] = lhs
                        rhs_all[i,:] = rhs
                        pbar.update(1)
                pbar.close()
                
                # Cleanup takes a surprisingly long time       
                print("Cleaning up workers...")
            print("Cleaned up workers")
        else:
            for i, (deformation_per_pixel, (date_ref, date_sec)) in tqdm.tqdm(enumerate(zip(deformations, dates)), total=n):
                _, rhs, lhs = _process_deformations(self, i, deformation_per_pixel, date_ref, date_sec)
                lhs_all[i,:] = lhs
                rhs_all[i,:] = rhs
                
        print("Solving equations")
        if only_solve_E:
            print("ONLY SOLVING E")
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
            # Scale answer according to Stefan-equation to time of calibration measurement.
            alt_pred = sol[0, :] * self.calib_root_ddt
        elif self.rtype == ReSALT_Type.SCReSALT_NS:
            alt_pred = []
            for m in sol[0,:]:
                # TODO: could also search discretized answers in self.sdi
                alt_hat = self.smm.thaw_depth_from_addt_m(self.calib_ddt, m)
                alt_pred.append(alt_hat)
            alt_pred = np.array(alt_pred)
        elif self.rtype == ReSALT_Type.LIU_SCHAEFER:
            E_idx = 0 if only_solve_E else 1
            E = sol[E_idx, :]
            if self.calib_idx is not None:
                # LIU_SCHAEFER is solved by finding delta E with respect to the calibration node at the end of the season.
                # "End of season" could be interpreted in two ways:
                # 1. When DDT is maximized or stops increasing
                # 2. When measurements of thaw depth were taken
                # Different results arise depending on which definition is taken. The following text
                # from "Remotely Sensed Active Layer Thickness (ReSALT) at Barrow, Alaska Using Interferometric Synthetic Aperture Radar" suggests to use definition 2:
                # "We calculated the long-term average of the in situ ALT measurements at this node and estimated the expected seasonal deformation using Equation (3) below. We then assumed that this represented a known deformation and used this location as the reference point to calibrate the subsidence for the entire domain."
                
                # Definition 2:
                calib_E = self.calib_deformation / self.calib_root_ddt
                # Definition 1:
                # eos_calib_alt = self.calib_alt * 1.0/self.calib_root_ddt
                # eos_calib_def = self.smm.deformation_from_alt(eos_calib_alt)
                # calib_E = eos_calib_def / 1.0 # 1.0 signifies end-of-season root DDT of 1.0
                
                delta_E = calib_E - E[self.calib_idx]
                E = E + delta_E
            alt_pred = []
            # Scale according to DDT at time of calibration measurement.
            E = E * self.calib_root_ddt
            for e in E:
                if e < 1e-3:
                    alt_pred.append(np.nan)
                    continue
                alt = self.smm.thaw_depth_from_deformation(e)
                alt_pred.append(alt)
            # If we used Definition 1, we would now scale by multiplying by self.calib_root_ddt
            alt_pred = np.array(alt_pred)
        else:
            raise ValueError()
        return alt_pred


def _process_deformations_wrapper(resalt_obj: ReSALT, i, deformation_per_pixel, date_ref, date_sec):
    try:
        result = _process_deformations(resalt_obj, i, deformation_per_pixel, date_ref, date_sec)
        return result
    except Exception as e:
        # Capture exception and the stacktrace
        return e, traceback.format_exc()


def _process_deformations(resalt_obj: ReSALT, i, deformation_per_pixel, date_ref, date_sec):
    delta_t_years = (date_ref - date_sec).days / 365
    norm_ddt_ref = get_norm_ddt(resalt_obj.df_temp, date_ref)
    norm_ddt_sec = get_norm_ddt(resalt_obj.df_temp, date_sec)
    
    if resalt_obj.rtype == ReSALT_Type.LIU_SCHAEFER:
        # Solve as deltas over calibration deformation. Later, this will be corrected.
        if resalt_obj.calib_idx is not None:
            delta = deformation_per_pixel[resalt_obj.calib_idx]
            lhs = deformation_per_pixel - delta
        else:
            lhs = deformation_per_pixel
        sqrt_norm_ddt_ref = np.sqrt(norm_ddt_ref)
        sqrt_norm_ddt_sec = np.sqrt(norm_ddt_sec)
        sqrt_addt_diff = sqrt_norm_ddt_ref - sqrt_norm_ddt_sec
        rhs = [delta_t_years, sqrt_addt_diff]
    elif resalt_obj.rtype == ReSALT_Type.SCReSALT:
        sqrt_norm_ddt_ref = np.sqrt(norm_ddt_ref)
        sqrt_norm_ddt_sec = np.sqrt(norm_ddt_sec)
        if resalt_obj.calib_alt:
            # Correct given deformations
            calib_ref_thaw_depth = resalt_obj.calib_alt * sqrt_norm_ddt_ref/resalt_obj.calib_root_ddt
            calib_sec_thaw_depth = resalt_obj.calib_alt * sqrt_norm_ddt_sec/resalt_obj.calib_root_ddt
            calib_ref_subsidence = resalt_obj.smm.deformation_from_thaw_depth(calib_ref_thaw_depth)
            calib_sec_subsidence = resalt_obj.smm.deformation_from_thaw_depth(calib_sec_thaw_depth)
            calib_def = calib_ref_subsidence - calib_sec_subsidence
            
            # Scale the deformations to align with the expected calibration deformation
            calib_node_delta = calib_def - deformation_per_pixel[resalt_obj.calib_idx]
            deformation_per_pixel = deformation_per_pixel + calib_node_delta
        
        lhs = scresalt_find_best_thaw_depth_difference(deformation_per_pixel, sqrt_norm_ddt_ref/sqrt_norm_ddt_sec, resalt_obj.smm, resalt_obj.upper_thaw_depth_limit, resalt_obj.N)
        
        # Sanity check
        expected_alt_diff = calib_ref_thaw_depth - calib_sec_thaw_depth
        err = abs(lhs[resalt_obj.calib_idx] - expected_alt_diff)
        assert err < 1e-3
        sqrt_addt_diff = sqrt_norm_ddt_ref - sqrt_norm_ddt_sec
        rhs = [sqrt_addt_diff]
    elif resalt_obj.rtype == ReSALT_Type.SCReSALT_NS:
        if resalt_obj.calib_alt:
            # Correct given deformations
            calib_ref_thaw_depth = resalt_obj.sdi.find_thaw_depth_for_integral(resalt_obj.calib_integral * norm_ddt_ref/resalt_obj.calib_ddt).h
            if calib_ref_thaw_depth is None:
                calib_ref_thaw_depth = resalt_obj.upper_thaw_depth_limit
            
            calib_sec_thaw_depth = resalt_obj.sdi.find_thaw_depth_for_integral(resalt_obj.calib_integral * norm_ddt_sec/resalt_obj.calib_ddt).h
            if calib_sec_thaw_depth is None:
                calib_sec_thaw_depth = resalt_obj.upper_thaw_depth_limit
            
            calib_ref_subsidence = resalt_obj.smm.deformation_from_thaw_depth(calib_ref_thaw_depth)
            calib_sec_subsidence = resalt_obj.smm.deformation_from_thaw_depth(calib_sec_thaw_depth)
            calib_def = calib_ref_subsidence - calib_sec_subsidence
            
            # Scale the deformations to align with the expected calibration deformation
            calib_node_delta = calib_def - deformation_per_pixel[resalt_obj.calib_idx]
            deformation_per_pixel = deformation_per_pixel + calib_node_delta
        
        tdp = scresalt_nonstefan_find_best_thaw_depth_pair(deformation_per_pixel, norm_ddt_ref, norm_ddt_sec, resalt_obj.smm, resalt_obj.sdi)
        # Sanity check
        expected_alt_diff = calib_ref_thaw_depth - calib_sec_thaw_depth
        diff = tdp[resalt_obj.calib_idx][1].h - tdp[resalt_obj.calib_idx][0].h
        err = abs(diff - expected_alt_diff)
        assert err < 3e-3
        lhs = []
        for (h1, h2) in tdp:
            lhi = resalt_obj.sdi.integrate(h1, h2)
            lhs.append(lhi)
        rhs = [norm_ddt_ref - norm_ddt_sec]
    else:
        raise ValueError("Unknown rtype:", resalt_obj.rtype)
        
    return (i, rhs, lhs)
        

def scresalt_find_best_thaw_depth_difference(deformation_per_pixel, sqrt_ddt_ratio, smm: SoilMoistureModel, upper_thaw_depth_limit, N):
    """
    Implements Algorithm 1 in the paper.
    """
    thaw_depth_differences, subsidence_differences = scresalt_generate_thaw_subsidence_mapping(sqrt_ddt_ratio, smm, upper_thaw_depth_limit, N)
    assert check_scresalt_requirement(subsidence_differences, sqrt_ddt_ratio**2), "SCReSALT requirement failed"
    sorter = np.arange(0, len(subsidence_differences), 1)
    if sqrt_ddt_ratio < 1:
        # Reverse will sort in increasing order
        sorter = sorter[::-1]
    best_matching_thaw_depth_differences = []
    for d in deformation_per_pixel:
        # TODO: interpolate here?
        idx = np.searchsorted(subsidence_differences, d, sorter=sorter)
        if idx == len(subsidence_differences):
            # This means the subsidence difference is too large. Substitute with largest possible pair.
            idx = len(subsidence_differences) - 1
        best_matching_thaw_depth_differences.append(thaw_depth_differences[sorter[idx]])
    return best_matching_thaw_depth_differences    
    
    
def scresalt_generate_thaw_subsidence_mapping(sqrt_ddt_ratio, smm: SoilMoistureModel, upper_thaw_depth_limit: float, N=1000):
    """
    Generates thaw depth pairs and computes corresponding subsidences. Returns thaw depth differences and corresponding subsidence differences.
    
    Args:
        upper_thaw_depth_limit: Limit the sampled thaw depths to be below this number.
        N: Number of thaw depth pairs to generate.
    """
    # assert sqrt_ddt_ratio != 1.0 # or something close...
    if sqrt_ddt_ratio > 1.0:
        # Under this condition, h1 < h2 so to make h2 upper bounded by `upper_thaw_depth_limit`,
        # we need h1 upper bounded by `upper_thaw_depth_limit/sqrt_ddt_ratio`
        # Under the opposite condition, h1 > h2 so `upper_thaw_depth_limit` is correct as it is.
        upper_thaw_depth_limit = upper_thaw_depth_limit/sqrt_ddt_ratio
    h1s = np.linspace(0.01, upper_thaw_depth_limit, num=N)
    h2s = sqrt_ddt_ratio * h1s
    subsidence_differences = []
    for h2, h1 in zip(h2s, h1s):
        sub = smm.deformation_from_thaw_depth(h2) - smm.deformation_from_thaw_depth(h1)
        subsidence_differences.append(sub)
    subsidence_differences = np.array(subsidence_differences)
    return h2s - h1s, subsidence_differences


def scresalt_nonstefan_find_best_thaw_depth_pair(deformation_per_pixel, ddt_ref, ddt_sec, smm: SoilMoistureModel, sdi: SoilDepthIntegration):
    """
    Implements modified Algorithm 1 presented in Appendix D.
    """
    ddt_ratio = ddt_ref/ddt_sec
    subsidence_differences, h1s, h2s = scresalt_nonstefan_generate_thaw_subsidence_mapping(ddt_ratio, smm, sdi)
    if not check_scresalt_requirement(subsidence_differences, ddt_ratio):
        print()
    # assert check_scresalt_requirement(subsidence_differences, ddt_ratio), "SCReSALT-NS requirement failed"
    sorter = np.arange(0, len(subsidence_differences), 1)
    if ddt_ratio < 1:
        # Reverse will sort in increasing order
        sorter = sorter[::-1]
    best_matching_thaw_depth_pairs = []
    for d in deformation_per_pixel:
        # TODO: interpolate here?
        idx = np.searchsorted(subsidence_differences, d, sorter=sorter)
        if idx == len(subsidence_differences):
            # This means the subsidence difference is too large. Substitute with largest possible pair.
            idx = len(subsidence_differences) - 1
        h1 = h1s[sorter[idx]]
        h2 = h2s[sorter[idx]]
        best_matching_thaw_depth_pairs.append((h1, h2))
    return best_matching_thaw_depth_pairs
    
    
def scresalt_nonstefan_generate_thaw_subsidence_mapping(ddt_ratio, smm: SoilMoistureModel, sdi: SoilDepthIntegration):
    h2s = []
    h1s = []
    # TODO: make `sdi` iterable?
    for i, (h1, h1_int) in enumerate(zip(sdi.h1s, sdi.all_h1_integrals)):
        h2_int = ddt_ratio * h1_int
        h2 = sdi.find_thaw_depth_for_integral(h2_int)
        if h2 is None:
            # The h2 will exceed the thaw depth limit.
            break
        assert h2.h <= sdi.upper_thaw_depth_limit
        h2s.append(h2)
        
        # Store the thaw depth and integration for `h1` as well
        h1 = SoilDepthIntegrationReturn(h1, h1_int)
        h1s.append(h1)
        
    subsidence_differences = []
    for h1, h2 in zip(h1s, h2s):
        sub = smm.deformation_from_thaw_depth(h2.h) - smm.deformation_from_thaw_depth(h1.h)
        subsidence_differences.append(sub)
    subsidence_differences = np.array(subsidence_differences)
    return subsidence_differences, h1s, h2s
    

def check_scresalt_requirement(subsidence_differences, ddt_ratio):
    """
    Checks that the generated subsidence differences grow (or decrease, if ddt_ratio < 1) strictly monotonically.
    """
    if ddt_ratio > 1:
        return all(subsidence_differences[i] < subsidence_differences[i+1] for i in range(len(subsidence_differences) - 1))
    else:
        return all(subsidence_differences[i] > subsidence_differences[i+1] for i in range(len(subsidence_differences) - 1))
        
