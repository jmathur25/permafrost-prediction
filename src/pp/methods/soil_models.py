"""
Contains soil models mentioned in the paper.
"""

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import root_scalar
from abc import ABC, abstractmethod

DENSITY_WATER = 0.997  # g/m^3
DENSITY_ICE = 0.9168  # g/cm^3

def liu_resalt_integrand(h):
    """
    Described in "Estimating 1992–2000 average active layer thickness on the Alaskan North Slope from remotely sensed surface subsidence".
    This characterizes the mixed soil model in the paper.
    """
    f_org = 0.0
    D_org = 0.18
    M_org = 30
    p_org_max = 140
    if h <= D_org:
        f_org = M_org / (p_org_max * D_org)
    k = 5.5
    D_root = 0.7
    p_org = k*M_org*np.exp(-k*h)/(1.0 - np.exp(-k*D_root))
    P_org = 0.9
    f_org = np.clip(p_org / p_org_max, 0.0, 1.0)
    P_mineral = 0.44
    P = (1.0 - f_org)*P_mineral + f_org*P_org
    S = 1.0
    return P * S

def invalid_smm(h):
    if h < 1e-3:
        return 0.9
    a = 10/9 - 0.01
    p = min(0.9, 1.0/(h + a))
    return p

class SoilMoistureModel(ABC):
    # TODO: rename ALT -> thaw depth
    @abstractmethod
    def deformation_from_thaw_depth(self, h: float) -> float:
        pass
    
    @abstractmethod
    def thaw_depth_from_deformation(self, deformation: float) -> float:
        pass
    
    @abstractmethod
    def porosity(self, h: float) -> float:
        pass

    def height_porosity_integrand(self, h) -> float:
        return self.porosity(h) * h
    
    def height_porosity_integration(self, h1, h2) -> float:
        integral, error = quad(self.height_porosity_integrand, h1, h2)
        assert error < 1e-5
        return integral
    
    def thaw_depth_from_addt_m(self, addt, m) -> float:
        def objective(x, target):
            integral, _ = quad(self.height_porosity_integrand, 0, x)
            return integral - target
        
        integral_val = addt * m
        result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
        assert result.converged
        return result.root
    
class SoilDepthIntegration:
    def __init__(self, smm: SoilMoistureModel, upper_thaw_depth_limit, N):
        self.upper_thaw_depth_limit = upper_thaw_depth_limit
        self.h1s = np.linspace(0.0, upper_thaw_depth_limit, num=N)
        h1_integrands = []
        for h1 in self.h1s:
            h1_integrands.append(smm.height_porosity_integrand(h1))
        self.all_h1_integrals = cumulative_trapezoid(h1_integrands, self.h1s, initial=0)
        
    def find_thaw_depth_for_integral(self, integral):
        idx = np.searchsorted(self.all_h1_integrals, integral)
        if idx == len(self.all_h1_integrals):
            return None
        elif idx == 0:
            return SoilDepthIntegrationReturn(self.h1s[idx], self.all_h1_integrals[idx])
        left_idx = idx - 1
        # Linearly interpolate between thaw depths
        h1_left = self.h1s[left_idx]
        h1_right = self.h1s[idx]
        int_left = self.all_h1_integrals[left_idx]
        int_right = self.all_h1_integrals[idx]
        w_left = (int_right - integral) / (int_right - int_left)
        h_return = h1_left * w_left + h1_right * (1 - w_left)
        int_return = int_left * w_left + int_right * (1 - w_left)
        return SoilDepthIntegrationReturn(h_return, int_return)
    
    def integrate(self, sdir1: "SoilDepthIntegrationReturn", sdir2: "SoilDepthIntegrationReturn"):
        """
        Integrates depth porosity function from sdir1 thaw depth to sdir2.
        """
        return sdir2.integral - sdir1.integral
    
class SoilDepthIntegrationReturn:
    """
    Stores linearly interpolated thaw depth and the depth-porosity integral.
    """
    def __init__(self, h, integral):
        self.h = h
        self.integral = integral    

    
class LiuSMM(SoilMoistureModel):
    """
    Describes in Liu et al. (2012). Referred to as the 'mixed soil model' in the paper and in Liu's paper.
    """
    
    def __init__(self):
        pass
    
    
    def deformation_from_thaw_depth(self, h):
        # This is relatively slow and is why SCReSALT is much slower than ReSALT. It needs to make subsidence difference to
        # thaw depth difference tables for each Q (basically per interferogram). This is relatively slow, so SCReSALT
        # becomes slow. Precomputing/caching could significantly speed things up.
        integral, error = quad(liu_resalt_integrand, 0, h)
        assert error < 1e-5
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * integral


    def thaw_depth_from_deformation(self, deformation):
        assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
        integral_val = deformation * (DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE))

        # Define the function to find its root
        def objective(x, target):
            integral, _ = quad(liu_resalt_integrand, 0, x)
            return integral - target

        # TODO: adjust bracket??
        result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
        assert result.converged
        return result.root
    
    def porosity(self, h):
        return liu_resalt_integrand(h)
    
    
class ChenSMM(SoilMoistureModel):
    """
    Not used in the paper. From: "Retrieval of Permafrost Active Layer Properties Using Time-Series P-Band Radar Observations" by Chen et al.
    """
    
    def __init__(self):
        pass
    
    
    def deformation_from_thaw_depth(self, h):
        integral, error = quad(self.porosity, 0, h)
        assert error < 1e-5
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * integral
        
        
    def thaw_depth_from_deformation(self, deformation):
        assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
        integral_val = deformation * (DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE))

        # Define the function to find its root
        def objective(x, target):
            integral, _ = quad(self.porosity, 0, x)
            return integral - target

        result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
        assert result.converged
        return result.root
    
    def som(self, h):
        om_neg_inf = 0.8
        om_inf = 0.05
        B = 50
        z_org = 0.15
        som = om_neg_inf + (om_inf - om_neg_inf) / (1 + np.exp(-B*(h - z_org)))
        return som
    
    def porosity(self, h):
        som = self.som(h)
        p_bm = 1.5
        r = 0.05
        p_b = p_bm * np.exp(-r * som)
        fc = 0.9887 * np.exp(-5.512 * p_b)
        p_m =  2.65
        p_s = 1.8
        p_f = 0.6
        porosity = 1 - p_b / p_m * (1 - som) # - p_b / p_s * som * (1 - fc) - p_b / p_f * som * fc
        return porosity
    

class ConstantWaterSMM(SoilMoistureModel):
    """
    Simple soil model with a constant porosity.
    """
    
    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p
    
    def deformation_from_thaw_depth(self, h: float) -> float:
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * h * self.p
    
    def thaw_depth_from_deformation(self, deformation: float) -> float:
        return deformation * DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE) / self.p
    
    def porosity(self, h):
        return self.p
    
class SCReSALT_Invalid_SMM(SoilMoistureModel):
    """
    The counterexample soil model derived in the appendix. This fails SCReSALT's requirement.
    """
    
    def __init__(self):
        x1 = 0.01
        x2 = 0.04

        f1_slope = 0.7/10
        f3_slope = 0.1/10

        y1 = 1/f3_slope*f1_slope*x1
        print("Transitions from 1st slope to 2nd slope between", x2, "and", y1)
        assert y1 > x2
        F1 = _Linear(f1_slope, 0.0)

        def solve_quadratic_and_align_funcs(x2, y1, F1, f3_slope):
            a = (F1.m - f3_slope)/(2*(x2-y1))
            b = F1.m - 2*a*x2
            c = F1(x2) - a*(x2**2) - b*x2
            F2 = _Quadratic(a, b, c)
            
            f3_start = F2(y1) - f3_slope * y1
            F3 = _Linear(f3_slope, f3_start)
            return F2, F3

        F2, F3 = solve_quadratic_and_align_funcs(x2, y1, F1, f3_slope)
        
        self.F1 = F1
        self.F2 = F2
        self.F3 = F3
        self.x2 = x2
        self.y1 = y1

        print("Linear from", x1, self.x2, "with", self.F1.m, self.F1.b)
        print("Quad from", self.x2, y1, "with", self.F2.a, self.F2.b, self.F2.c)
        print("Linear from", y1, "with", self.F3.m, self.F3.b)
        
    def deformation_from_thaw_depth(self, h: float) -> float:
        if h <= self.x2:
            return self.F1(h)
        elif h <= self.y1:
            return self.F2(h)
        else:
            return self.F3(h)
    
    def thaw_depth_from_deformation(self, deformation: float) -> float:
        raise ValueError("unimplemented")
    
    def porosity(self, h: float):
        if h <= self.x2:
            v = self.F1.get_derivative(h)
        elif h <= self.y1:
            v = self.F2.get_derivative(h)
        else:
            v = self.F3.get_derivative(h)
        return DENSITY_ICE * v / (DENSITY_WATER - DENSITY_ICE)
    
class _Linear:
    def __init__(self, m, b):
        self.m = m
        self.b = b
        
    def __call__(self, x):
        return self.m * x + self.b
    
    def get_derivative(self, x):
        return self.m
    

class _Quadratic:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        
    def __call__(self, x):
        return self.a * (x**2) + self.b * x + self.c
    
    def get_derivative(self, x):
        return 2*self.a*x + self.b
