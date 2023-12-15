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

def invalid_smm(z):
    if z < 1e-3:
        return 0.9
    a = 10/9 - 0.01
    p = min(0.9, 1.0/(z + a))
    return p

class SoilMoistureModel(ABC):
    @abstractmethod
    def deformation_from_alt(self, alt: float) -> float:
        pass
    
    @abstractmethod
    def alt_from_deformation(self, deformation: float) -> float:
        pass
    
    @abstractmethod
    def porosity(self, z: float) -> float:
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
    
    def porosity(self, z):
        return liu_resalt_integrand(z)
    
    
class ChenSMM(SoilMoistureModel):
    def __init__(self):
        pass
    
    
    def deformation_from_alt(self, alt):
        integral, error = quad(self.porosity, 0, alt)
        assert error < 1e-5
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * integral
        
        
    def alt_from_deformation(self, deformation):
        assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
        integral_val = deformation * (DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE))

        # Define the function to find its root
        def objective(x, target):
            integral, _ = quad(self.porosity, 0, x)
            return integral - target

        result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
        assert result.converged
        return result.root
    
    def som(self, z):
        om_neg_inf = 0.8
        om_inf = 0.05
        B = 50
        z_org = 0.15
        som = om_neg_inf + (om_inf - om_neg_inf) / (1 + np.exp(-B*(z - z_org)))
        return som
    
    def porosity(self, z):
        som = self.som(z)
        p_bm = 1.5
        r = 0.05
        p_b = p_bm * np.exp(-r * som)
        fc = 0.9887 * np.exp(-5.512 * p_b)
        p_m =  2.65
        p_s = 1.8
        p_f = 0.6
        porosity = 1 - p_b / p_m * (1 - som) # - p_b / p_s * som * (1 - fc) - p_b / p_f * som * fc
        return porosity
        

class SCReSALT_Invalid_SMM2(SoilMoistureModel):
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
        # integral, error = quad(invalid_smm, 0, alt)
        # assert error < 1e-5
        # return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * integral
        return np.clip(np.log(alt*100) + 0.1, 0.0, 10.0)


    def alt_from_deformation(self, deformation):
        raise ValueError()
        # assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
        # integral_val = deformation * (DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE))

        # # Define the function to find its root
        # def objective(x, target):
        #     integral, _ = quad(invalid_smm, 0, x)
        #     return integral - target

        # result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
        # assert result.converged
        # return result.root
    
    def porosity(self, z):
        return invalid_smm(z)


class ConstantWaterSMM(SoilMoistureModel):
    def __init__(self, p: float):
        assert 0 <= p <= 1
        self.p = p
    
    def deformation_from_alt(self, alt: float) -> float:
        return (DENSITY_WATER - DENSITY_ICE) / DENSITY_ICE * alt * self.p
    
    def alt_from_deformation(self, deformation: float) -> float:
        return deformation * DENSITY_ICE / (DENSITY_WATER - DENSITY_ICE) / self.p
    
    def porosity(self, z):
        return self.p


class SCReSALT_Invalid_SMM(SoilMoistureModel):
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
        
    def deformation_from_alt(self, alt: float) -> float:
        if alt <= self.x2:
            return self.F1(alt)
        elif alt <= self.y1:
            return self.F2(alt)
        else:
            return self.F3(alt)
    
    def alt_from_deformation(self, deformation: float) -> float:
        raise ValueError("unimplemented")
    
    def porosity(self, z: float):
        if z <= self.x2:
            v = self.F1.get_derivative(z)
        elif z <= self.y1:
            v = self.F2.get_derivative(z)
        else:
            v = self.F3.get_derivative(z)
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
