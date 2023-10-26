
from datetime import datetime
import enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from methods.schaefer import get_norm_ddt
from methods.soil_models import SoilMoistureModel


class ReSALT_Type(enum.Enum):
    LIU_SCHAEFER = 0
    JATIN = 1


class ReSALT:
    def __init__(self, df_temp: pd.DataFrame, smm: SoilMoistureModel, rtype: ReSALT_Type, calib: Tuple[int, float]=None):
        """
        TODO: add support for gravel calibration point?
        """
        self.df_temp = df_temp
        if calib:
            self.calib_idx = calib[0]
            self.calib_deformation = calib[1]
        else:
            self.calib_idx = None
            self.calib_deformation = None
        self.smm = smm
        self.rtype = rtype
    

    def run_inversion(self, deformations: np.ndarray, dates: List[Tuple[datetime, datetime]]) -> np.ndarray:
        """
        deformations: 2D-array of (igram_idx, pixel) that stores deformation
        dates: list of dates of the igrams. It is assumed that the first index
            corresponds to the 'reference' and the second index to the 'secondary'. This means the deformation should be derived from:
            phase in the reference - phase in secondary.
        """
        n = deformations.shape[0]
        lhs_all = np.zeros((n, deformations.shape[1]))
        rhs_all = np.zeros((n, 2))
        for i, (deformation, (date_ref, date_sec)) in enumerate(zip(deformations, dates)):
            delta_t_years = (date_ref - date_sec).days / 365
            norm_ddt_ref = get_norm_ddt(self.df_temp, date_ref)
            norm_ddt_sec = get_norm_ddt(self.df_temp, date_sec)
            sqrt_addt_diff = np.sqrt(norm_ddt_ref) - np.sqrt(norm_ddt_sec)
            rhs = [delta_t_years, sqrt_addt_diff]
            
            if self.calib_idx:
                # Set the deformation to a delta w.r.t calibration point
                delta = deformation[self.calib_idx]
                lhs = deformation - delta
            else:
                lhs = deformation
            lhs_all[i,:] = lhs
            rhs_all[i,:] = rhs
        
        print("Solving equations")
        rhs_all = rhs_all[:, [1]]
        rhs_pi = np.linalg.pinv(rhs_all)
        sol = rhs_pi @ lhs_all

        # R = sol[0, :]
        E = sol[0, :]
        
        # Calibrate wrt known deformation
        if self.calib_idx:
            delta_E = self.calib_deformation - E[self.calib_idx]
            E = E + delta_E

        print("AVG E", np.mean(E))

        alt_pred = []
        for e in E:
            if e < 1e-3:
                alt_pred.append(np.nan)
                continue
            alt = self.smm.alt_from_deformation(e)
            alt_pred.append(alt)

        alt_pred = np.array(alt_pred)
        return alt_pred

        
    