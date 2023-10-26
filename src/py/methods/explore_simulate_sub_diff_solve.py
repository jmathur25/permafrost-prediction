# %%
from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import alt_to_surface_deformation

# %%
sqrt_ddt_ratio = 1.1
h1s = np.linspace(0.01, 0.09, num=100)
h2s = sqrt_ddt_ratio * h1s
subsidences = []
for h2, h1 in zip(h2s, h1s):
    sub = alt_to_surface_deformation(h2) - alt_to_surface_deformation(h1)
    subsidences.append(sub)
subsidences = np.array(subsidences)

# %%
plt.scatter(subsidences, h2s)
plt.xlabel("def per pixel")
plt.ylabel("ALT2")

# %%
hs = np.linspace(0.01, 0.4, num=100)
sub = [alt_to_surface_deformation(h) for h in hs]
plt.scatter(hs, sub)

# %%
