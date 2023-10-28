# %%
from matplotlib import pyplot as plt
import numpy as np

import sys
sys.path.append("/permafrost-prediction/src/py")
from methods.soil_models import LiuSMM, ConstantWaterSMM

# %%
smm = LiuSMM()
ddt_ref = 20
ddt_sec = 10
sqrt_ddt_ratio = np.sqrt(ddt_ref)/np.sqrt(ddt_sec)
h1s = np.linspace(0.01, 0.09, num=100)
h2s = sqrt_ddt_ratio * h1s
subsidences = []
for h2, h1 in zip(h2s, h1s):
    sub = smm.deformation_from_alt(h2) - smm.deformation_from_alt(h1)
    subsidences.append(sub)
subsidences = np.array(subsidences)

# %%
plt.scatter(subsidences, h2s-h1s)
plt.xlabel("def per pixel")
plt.ylabel("ALT2-ALT1")

# %%
hs = np.linspace(0.01, 0.4, num=100)
sub = [smm.deformation_from_alt(h) for h in hs]
plt.scatter(hs, sub)
plt.xlabel("ALT")
plt.ylabel("subsidence")

# %%
