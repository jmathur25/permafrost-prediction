# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import numpy as np

sys.path.append("/permafrost-prediction/src/py")
from methods.simulate_sub_diff_solve import generate_h1_h2_subs
from methods.soil_models import LiuSMM, liu_resalt_integrand, ConstantWaterSMM, SCReSALT_Invalid_SMM

# %%
plt.rcParams.update({'font.size': 15})  # This will change the font size globally

# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })
# plt.figure()
# plt.xlabel(r'This is a test: $\alpha$')
# plt.show()

# %%

liu_smm = LiuSMM()
inv_smm = SCReSALT_Invalid_SMM()
const_smm = ConstantWaterSMM(0.75)

# %%
zs = np.linspace(0.0, 1.0, 1000)
ps_mixed_model = np.array([liu_smm.porosity(z) for z in zs])
ps_constant = np.array([const_smm.porosity(z) for z in zs])
subs_mixed_model = np.array([liu_smm.deformation_from_alt(z) for z in zs])
subs_const_water_model = np.array([const_smm.deformation_from_alt(z) for z in zs])

# Create a figure with two subplots (1 row, 2 columns)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# TODO: plot limit to 0.5m, and figure out weird curve for green porosity
ax1.plot(zs, ps_mixed_model, color='b', label='Mixed Soil Model')
ax1.plot(zs, ps_constant, color='r', label='Constant Porosity Model')
ax1.set_xlabel("ALT (m)")
ax1.set_ylabel("Porosity")
ax1.set_xlim(0.0, 0.5)

# Second plot: ALT vs Subsidence
ax2.plot(zs, subs_mixed_model, color='b', label='Mixed Soil Model')
ax2.plot(zs, subs_const_water_model, color='r', label='Constant Porosity Model')
ax2.set_xlabel("ALT (m)")
ax2.set_ylabel("Subsidence (m)")
ax2.set_xlim(0.0, 0.5)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

def plot_sqrt_ddt(smm, color, sqrt_ddt_ratio, ax, ylim=None):
    sqrt_ddt_ref = 15
    sqrt_ddt_sec = sqrt_ddt_ref/sqrt_ddt_ratio
    upper_alt_limit = zs[-1]/sqrt_ddt_ratio
    h1s, h2s, subsidences = generate_h1_h2_subs(sqrt_ddt_ref, sqrt_ddt_sec, smm, upper_alt_limit, N=100)
    ax.plot(h2s - h1s, subsidences, color=color)
    ax.set_title(fr"ALT difference vs differential subsidence for $\sqrt{{\frac{{ADDT_{{t_i}}}}{{ADDT_{{t_j}}}}}} = {sqrt_ddt_ratio}$")
    ax.set_xlabel(r"$ALT_i - ALT_j \, (m)$")  # Thin space
    ax.set_ylabel("Differential Subsidence (m)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(0.0, 0.3)

sqrt_ddt_ratio = 3.5
plot_sqrt_ddt(liu_smm, 'b', sqrt_ddt_ratio, ax3, ylim=(0.0, 0.05))
plot_sqrt_ddt(const_smm, 'r', sqrt_ddt_ratio, ax3)

sqrt_ddt_ratio = 1.5
plot_sqrt_ddt(liu_smm, 'b', sqrt_ddt_ratio, ax4, ylim=(0.0, 0.05))
plot_sqrt_ddt(const_smm, 'r', sqrt_ddt_ratio, ax4)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplots to fit into the figure area.
plt.show()
fig.savefig("alt_porosity_subsidence_plots.png")

# %%
for z in zs:
    p = inv_smm.porosity(z)
# raise ValueError()
ps_inv_model = np.array([inv_smm.porosity(z) for z in zs])
subs_inv_model = np.array([inv_smm.deformation_from_alt(z) for z in zs])
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

color = 'purple'

ax1.plot(zs, ps_inv_model, color=color, label='Invalid Model')
ax1.set_xlabel("ALT (m)")
ax1.set_ylabel("Porosity")
ax1.set_xlim(0.0, 0.5)

# Second plot: ALT vs Subsidence
ax2.plot(zs, subs_inv_model, color=color, label='Invalid Model')
ax2.set_xlabel("ALT (m)")
ax2.set_ylabel("Subsidence (m)")
ax2.set_xlim(0.0, 0.5)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plot_sqrt_ddt(inv_smm, color, 3.5, ax3, ylim=(0.0, 0.005))
plot_sqrt_ddt(inv_smm, color, 1.5, ax4)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplots to fit into the figure area.
plt.show()
fig.savefig("alt_porosity_subsidence_plots_invalid_smm.png")

# %%
liu_smm = LiuSMM()

# %%
liu_smm.deformation_from_alt()
