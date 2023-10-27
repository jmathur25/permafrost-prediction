# %%
%load_ext autoreload
%autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

sys.path.append("..")
from methods.utils import compute_stats
from methods.soil_models import liu_deformation_from_alt, liu_alt_from_deformation


# %%
sub_pred = np.load("../../../subsidence.npy")
alt_pred = np.load("../../../alt_preds.npy")
alt_gt = np.load("../../../alt_gt.npy")
df_alt_gt = pd.read_csv("../../../df_alt_gt.csv")

# %%
plt.scatter(alt_pred, alt_gt)
# %%
mp = alt_gt.mean()
rmse_mean = np.sqrt(np.square(mp - alt_gt).mean())
rmse_preds = np.sqrt(np.square(alt_pred - alt_gt).mean())
print(rmse_mean, rmse_preds)

# %%
sub_gt = np.array([liu_deformation_from_alt(alt) for alt in alt_gt])

# %%
plt.scatter(sub_pred, sub_gt)
plt.xlabel('Subsidence pred')
plt.ylabel('Subsidence gt')

# %%
mp = sub_gt.mean()
rmse_mean = np.sqrt(np.square(mp - sub_gt).mean())
rmse_preds = np.sqrt(np.square(sub_pred - sub_gt).mean())
print(rmse_mean, rmse_preds)

# %%
diff = mp - np.mean(sub_pred)
sub_pred += diff
# %%
alt_pred2 = np.array([liu_alt_from_deformation(d) for d in sub_pred])
mp = alt_gt.mean()
rmse_mean = np.sqrt(np.square(mp - alt_gt).mean())
rmse_preds = np.sqrt(np.square(alt_pred2 - alt_gt).mean())
print(rmse_mean, rmse_preds)

# %%
compute_stats(alt_pred2, alt_gt, df_alt_gt['point_id'])

# %%
