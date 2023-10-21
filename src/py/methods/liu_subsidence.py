
# %%
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# %%
# best_fit_subsidence = np.array([7.0, 2.1, 1.3, 7.4, 4.3, 2.9, 4.1, 1.3, 1.1, 2.4, 2.6, 3.5])
best_extended_fit_sub = np.array([12.5, 2.9, 3.1, 10.6, 8.5, 4.4, 5.3, 2.0, 1.8, 4.8, 4.4, 9.1, 5.8])
modeled_pore_ice_sub = np.array([3.1, 2.6, 2.5, 2.5, 2.6, 2.6, 3.0, 3.0, 2.8, 3.0, 2.7, 2.9, 2.8])

# %%
pearson_r, _ = pearsonr(best_extended_fit_sub, modeled_pore_ice_sub)
print(pearson_r)


# %%
plt.scatter(best_extended_fit_sub, modeled_pore_ice_sub)
plt.xlabel("GPS-IR sub")
plt.ylabel("Model pore ice sub")

# %%
mae = np.mean(np.abs(best_extended_fit_sub - modeled_pore_ice_sub))
print("mae", mae)

# %%
