# %%
%load_ext autoreload
%autoreload 2
# %%
from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import DENSITY_ICE, DENSITY_WATER, SCReSALT_Invalid_SMM

# %%
x1 = 0.01
x2 = 0.05

f1_slope = 0.7/10
f3_slope = 0.1/10

y1 = 1/f3_slope*f1_slope*x1
y2 = 0.5
assert y1 <= y2

class Linear:
    def __init__(self, m, b):
        self.m = m
        self.b = b
        
    def __call__(self, x):
        return self.m * x + self.b
    
    def get_derivative(self, x):
        return self.m
    

class Quadratic:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        
    def __call__(self, x):
        return self.a * (x**2) + self.b * x + self.c
    
    def get_derivative(self, x):
        return 2*self.a*x + self.b
    
F1 = Linear(f1_slope, 0.0)

def solve_quadratic_and_align_funcs(x2, y1, F1, f3_slope):
    a = (F1.m - f3_slope)/(2*(x2-y1))
    b = F1.m - 2*a*x2
    c = F1(x2) - a*(x2**2) - b*x2
    F2 = Quadratic(a, b, c)
    
    f3_start = F2(y1) - f3_slope * y1
    F3 = Linear(f3_slope, f3_start)
    return F2, F3

F2, F3 = solve_quadratic_and_align_funcs(x2, y1, F1, f3_slope)

# %%
# We want to show that for any c such that x1 < cx1 <= x2, f1(cx1) - f1(x1) = f2(cy1) - f2(y1)

cs = np.linspace(1.0, x2/x1, 100)
f1s = []
f3s = []
for c in cs:
    f1s.append(F1(c*x1)-F1(x1))
    f3s.append(F3(c*y1)-F3(y1))
plt.scatter(f1s, f3s)

# %%
xs = np.linspace(x1, y2, 1000)
sc_inv_smm = SCReSALT_Invalid_SMM()
ys = []
for x in xs:
    sub = sc_inv_smm.deformation_from_alt(x)
    ys.append(sub)
plt.scatter(xs, ys)
plt.xlabel("ALT (m)")
plt.ylabel("Subsidence (m)")

# %%
ps = []
for x in xs:
    if x <= x2:
        d = F1.get_derivative(x)
    elif x <= y1:
        d = F2.get_derivative(x)
    else:
        d = F3.get_derivative(x)
    p = DENSITY_ICE * d / (DENSITY_WATER - DENSITY_ICE)
    ps.append(p)
plt.scatter(xs, ps)

# %%
