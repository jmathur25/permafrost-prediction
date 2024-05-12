"""
Creates the figures for soil models.
"""

# %%
%load_ext autoreload
%autoreload 2

# %%
import sys
from matplotlib import pyplot as plt
import numpy as np


sys.path.append("/permafrost-prediction/src")
from pp.methods.resalt import scresalt_generate_thaw_subsidence_mapping, scresalt_nonstefan_generate_thaw_subsidence_mapping
from pp.methods.soil_models import LiuSMM, SoilDepthIntegration, SoilMoistureModel, liu_resalt_integrand, ConstantWaterSMM, SCReSALT_Invalid_SMM, ChenSMM

# %%
plt.rcParams.update({'font.size': 15})

# %%

liu_smm = LiuSMM()
inv_smm = SCReSALT_Invalid_SMM()
const_smm = ConstantWaterSMM(0.75)

MAX_THAW_DEPTH = 2.0
N = 1000

# %%
def plot_sqrt_ddt(smm, color, sqrt_ddt_ratio, ax, ylim=None, linestyle=None):
    thaw_depth_differences, subsidence_differences = scresalt_generate_thaw_subsidence_mapping(sqrt_ddt_ratio, smm, MAX_THAW_DEPTH, N)
    ax.plot(thaw_depth_differences, subsidence_differences, color=color, linestyle=linestyle)
    ax.set_title(fr"Subsidence Difference vs Thaw Depth Difference for $K = {sqrt_ddt_ratio}$")
    ax.set_xlabel(r"$h_{t_i} - h_{t_j} \, (m)$")  # Thin space
    ax.set_ylabel(r"$\delta_{t_i} - \delta_{t_j} \, (m)$")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(0.0, 0.3)

# %%
zs = np.linspace(0.0, MAX_THAW_DEPTH, N)
ps_mixed_model = np.array([liu_smm.porosity(z) for z in zs])
ps_constant = np.array([const_smm.porosity(z) for z in zs])
subs_mixed_model = np.array([liu_smm.deformation_from_thaw_depth(z) for z in zs])
subs_const_water_model = np.array([const_smm.deformation_from_thaw_depth(z) for z in zs])

# Create a figure with 2 rows, 2 columns
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# First plot
ax1.plot(zs, ps_mixed_model, color='b', label='Mixed Soil Model')
ax1.plot(zs, ps_constant, color='r', linestyle='dashed', label='Constant Porosity Model')
ax1.set_xlabel("Thaw Depth (m)")
ax1.set_ylabel("Porosity")
ax1.set_xlim(0.0, 0.5)
ax1.set_title("Porosity vs Thaw Depth")

# Second plot
ax2.plot(zs, subs_mixed_model, color='b', label='Mixed Soil Model')
ax2.plot(zs, subs_const_water_model, color='r', linestyle='dashed', label='Constant Porosity Model')
ax2.set_xlabel("Thaw Depth (m)")
ax2.set_ylabel("Subsidence (m)")
ax2.set_xlim(0.0, 0.5)
ax2.set_title("Subsidence vs Thaw Depth")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

# Third plot
sqrt_ddt_ratio = 3.5
plot_sqrt_ddt(liu_smm, 'b', sqrt_ddt_ratio, ax3, ylim=(0.0, 0.05))
plot_sqrt_ddt(const_smm, 'r', sqrt_ddt_ratio, ax3, linestyle='dashed')

# Fourth plot
sqrt_ddt_ratio = 1.5
plot_sqrt_ddt(liu_smm, 'b', sqrt_ddt_ratio, ax4, ylim=(0.0, 0.05))
plot_sqrt_ddt(const_smm, 'r', sqrt_ddt_ratio, ax4, linestyle='dashed')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplots to fit into the figure area.
plt.show()
fig.savefig("thaw_depth_porosity_subsidence_plots.png")

# %%
ps_inv_model = np.array([inv_smm.porosity(z) for z in zs])
subs_inv_model = np.array([inv_smm.deformation_from_thaw_depth(z) for z in zs])
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

color = 'purple'

ax1.plot(zs, ps_inv_model, color=color, label='Invalid Model')
ax1.set_xlabel("Thaw Depth (m)")
ax1.set_ylabel("Porosity")
ax1.set_xlim(0.0, 0.5)
ax1.set_ylim(0.0, 1.0)
ax1.set_title("Porosity vs Thaw Depth")

# Second plot: ALT vs Subsidence
ax2.plot(zs, subs_inv_model, color=color, label='Invalid Model')
ax2.set_xlabel("Thaw Depth (m)")
ax2.set_ylabel("Subsidence (m)")
ax2.set_xlim(0.0, 0.5)
ax2.set_title("Subsidence vs Thaw Depth")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plot_sqrt_ddt(inv_smm, color, 3.5, ax3, ylim=(0.0, 0.005))
plot_sqrt_ddt(inv_smm, color, 1.5, ax4, ylim=(0.0, 0.005))

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplots to fit into the figure area.
plt.show()
fig.savefig("alt_porosity_subsidence_plots_invalid_smm.png")


# %%
liu_sdi = SoilDepthIntegration(liu_smm, MAX_THAW_DEPTH, N)
const_sdi = SoilDepthIntegration(const_smm, MAX_THAW_DEPTH, N)

# %%
def plot_ddt(smm: SoilMoistureModel, sdi: SoilDepthIntegration, ddt_ratio, ax, linestyle=None, color=None, label=None):
    subsidence_differences, h1s, h2s = scresalt_nonstefan_generate_thaw_subsidence_mapping(ddt_ratio, smm, sdi)
    h1s = [h1.h for h1 in h1s]
    ax.plot(h1s, subsidence_differences, color=color, linestyle=linestyle, label=label)
    ax.set_title(fr"Subsidence Difference vs Thaw Depth for DDT ratio $ = {ddt_ratio}$")
    ax.set_xlabel(r"$h_{t_j} \, (m)$")  # Thin space
    ax.set_ylabel(r"$\delta_{t_i} - \delta_{t_j} \, (m)$")
    
# Create a figure with 1 rows, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))

plot_ddt(liu_smm, liu_sdi, 3.5, ax1, color='b', label='Mixed Soil Model')
plot_ddt(liu_smm, liu_sdi, 1.5, ax2, color='b', label='Mixed Soil Model')

plot_ddt(const_smm, const_sdi, 3.5, ax1, linestyle='dashed', color='r', label='Constant Porosity Model')
plot_ddt(const_smm, const_sdi, 1.5, ax2, linestyle='dashed', color='r', label='Constant Porosity Model')

ax1.set_xlim(0.0, 0.5)
ax1.set_ylim(0.0, 0.03)

ax2.set_xlim(0.0, 0.5)
ax2.set_ylim(0.0, 0.03)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust subplots to fit into the figure area.
plt.show()
fig.savefig("scresalt_nonstefan.png")

# %%
