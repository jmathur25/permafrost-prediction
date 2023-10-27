
from datetime import datetime
import enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from methods.schaefer import get_norm_ddt
from methods.soil_models import SoilMoistureModel
from methods.simulate_sub_diff_solve import find_best_alt_diff


class ReSALT_Type(enum.Enum):
    LIU_SCHAEFER = 0
    JATIN = 1


class ReSALT:
    def __init__(self, df_temp: pd.DataFrame, smm: SoilMoistureModel, calib_idx: int, calib_deformation: float, rtype: ReSALT_Type):
        """
        TODO: add support for gravel calibration point?
        """
        self.df_temp = df_temp
        self.calib_idx = calib_idx
        self.calib_deformation = calib_deformation
        self.calib_alt = smm.alt_from_deformation(self.calib_deformation)
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
        for i, (deformation_per_pixel, (date_ref, date_sec)) in enumerate(zip(deformations, dates)):
            delta_t_years = (date_ref - date_sec).days / 365
            norm_ddt_ref = get_norm_ddt(self.df_temp, date_ref)
            norm_ddt_sec = get_norm_ddt(self.df_temp, date_sec)
            sqrt_norm_ddt_ref = np.sqrt(norm_ddt_ref)
            sqrt_norm_ddt_sec = np.sqrt(norm_ddt_sec)
            sqrt_addt_diff = sqrt_norm_ddt_ref - sqrt_norm_ddt_sec
            rhs = [delta_t_years, sqrt_addt_diff]
            
            if self.rtype == ReSALT_Type.LIU_SCHAEFER:
                # Solve as deltas over calibration deformation
                delta = deformation_per_pixel[self.calib_idx]
                lhs = deformation_per_pixel - delta
            elif self.rtype == ReSALT_Type.JATIN:
                # TODO: easily scales to calib being a non-thaw point like gravel, as all we ultimately do
                # is calculate subsidence within an image?
                expected_date_ref_alt = self.calib_alt * sqrt_norm_ddt_ref
                expected_date_ref_sec = self.calib_alt * sqrt_norm_ddt_sec
                expected_deformation_ref = self.smm.deformation_from_alt(expected_date_ref_alt)
                expected_deformation_sec = self.smm.deformation_from_alt(expected_date_ref_sec)
                expected_calib_def = expected_deformation_ref - expected_deformation_sec
                
                # Scale the deformations to align with the expected calibration deformation
                calib_node_delta = expected_calib_def - deformation_per_pixel[self.calib_idx]
                deformation_per_pixel = deformation_per_pixel + calib_node_delta
                
                lhs = find_best_alt_diff(deformation_per_pixel, sqrt_norm_ddt_ref, sqrt_norm_ddt_sec, self.smm)
            else:
                raise ValueError()
                
                
            lhs_all[i,:] = lhs
            rhs_all[i,:] = rhs
        
        print("Solving equations")
        if only_solve_E or self.rtype == ReSALT_Type.JATIN:
            # Jatin mode cannot solve for R
            rhs_all = rhs_all[:, [1]]
        rhs_pi = np.linalg.pinv(rhs_all)
        sol = rhs_pi @ lhs_all
        
        # if self.rtype == ReSALT_Type.JATIN:
        #     # Jatin mode can induce nans
        #     nans = np.argwhere(np.isnan(alt_pred))
        #     if len(nans) > 0:
        #         resolved = 0
        #         for i in nans[:,0]:
        #             lhs_i = lhs_all[:,i]
        #             nan_mask = np.isnan(lhs_i)
        #             if np.mean(nan_mask) > 0.5:
        #                 continue
        #             not_nan_mask = ~nan_mask
        #             alt_i = np.linalg.pinv(rhs_all[not_nan_mask]) @ lhs_i[not_nan_mask]
        #             alt_pred[i] = alt_i
        #             resolved += 1
        #         print("Number of pixels with nans in least-squares inversion:", len(nans))
        assert np.isnan(sol).sum() == 0

        if self.rtype == ReSALT_Type.LIU_SCHAEFER:
            E_idx = 0 if only_solve_E else 1
            E = sol[E_idx, :]
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
        elif self.rtype == ReSALT_Type.JATIN:
            alt_pred = sol[0, :]
        else:
            raise ValueError()
        return alt_pred

        
    