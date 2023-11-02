import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from abc import ABC, abstractmethod


DENSITY_WATER = 0.997  # g/m^3
DENSITY_ICE = 0.9168  # g/cm^3

# TODO: isn't there an initial thing??
# def resalt_integrand(z):
#     po = 0.9  # 1.0
#     pf = 0.45  # 0.454
#     # Chosen to make the VWC ~0.62 for an ALT of 0.36 cm (see Table 2)
#     # because paper reported this general average and this ALT average
#     # for CALM. Also the paper they cite [21?] has k=5.5
#     k = 5.5  # 5.239
#     return pf + (po - pf) * np.exp(-k * z)


def liu_resalt_integrand(z):
    f_org = 0.0
    D_org = 0.18
    M_org = 30
    p_org_max = 140
    if z <= D_org:
        f_org = M_org / (p_org_max * D_org)
    k = 5.5
    D_root = 0.7
    p_org = k*M_org*np.exp(-k*z)/(1.0 - np.exp(-k*D_root))
    P_org = 0.9
    f_org = np.clip(p_org / p_org_max, 0.0, 1.0)
    P_mineral = 0.44
    P = (1.0 - f_org)*P_mineral + f_org*P_org
    S = 1.0
    return P * S

class SoilMoistureModel(ABC):
    @abstractmethod
    def deformation_from_alt(self, alt: float) -> float:
        pass
    
    @abstractmethod
    def alt_from_deformation(self, deformation: float) -> float:
        pass


class LiuSMM(SoilMoistureModel):
    def __init__(self):
        pass
    
    
    def deformation_from_alt(self, alt):
        # paper assumes exponential decay from 90% porosity to 45% porosity
        # in general:
        # P(z) = P_f + (Po - Pf)*e^(-kz)
        # where P(f) = final porosity
        #       P(o) = intial porosity
        #       z is rate of exponential decay
        # Without reading citation, let us assume k = 1
        # Definite integral is now: https://www.wolframalpha.com/input?i=integrate+a+%2B+be%5E%28-kz%29+dz+from+0+to+x
        integral, error = quad(liu_resalt_integrand, 0, alt)
        assert error < 1e-5
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * integral


    def alt_from_deformation(self, deformation):
        assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
        integral_val = deformation * (DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE))

        # Define the function to find its root
        def objective(x, target):
            integral, _ = quad(liu_resalt_integrand, 0, x)
            return integral - target

        result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
        assert result.converged
        return result.root


class ConstantWaterSMM(SoilMoistureModel):
    def __init__(self, s: float):
        assert 0 <= s <= 1
        self.s = s
    
    def deformation_from_alt(self, alt: float) -> float:
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * alt * self.s
    
    def alt_from_deformation(self, deformation: float) -> float:
        return deformation * DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE) / self.s
