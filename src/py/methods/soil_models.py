import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


def resalt_integrand(z):
    po = 0.9  # 1.0
    pf = 0.45  # 0.454
    # Chosen to make the VWC ~0.62 for an ALT of 0.36 cm (see Table 2)
    # because paper reported this general average and this ALT average
    # for CALM. Also the paper they cite [21?] has k=5.5
    k = 5.5  # 5.239
    return pf + (po - pf) * np.exp(-k * z)


def alt_to_surface_deformation(alt):
    # paper assumes exponential decay from 90% porosity to 45% porosity
    # in general:
    # P(z) = P_f + (Po - Pf)*e^(-kz)
    # where P(f) = final porosity
    #       P(o) = intial porosity
    #       z is rate of exponential decay
    # Without reading citation, let us assume k = 1
    # Definite integral is now: https://www.wolframalpha.com/input?i=integrate+a+%2B+be%5E%28-kz%29+dz+from+0+to+x
    pw = 0.997  # g/m^3
    pi = 0.9168  # g/cm^3
    integral, error = quad(resalt_integrand, 0, alt)
    assert error < 1e-5
    return (pw - pi) / pi * integral


def compute_alt_f_deformation(deformation):
    assert deformation > 0, "Must provide a positive deformation in order to compute ALT"
    pw = 0.997  # g/m^3
    pi = 0.9168  # g/cm^3
    integral_val = deformation * (pi / (pw - pi))

    # Define the function to find its root
    def objective(x, target):
        integral, _ = quad(resalt_integrand, 0, x)
        return integral - target

    result = root_scalar(objective, args=(integral_val,), bracket=[0, 10], method="brentq")
    assert result.converged
    return result.root
