import pyllama as ll
import numpy as np
import cholesteric as ch
import matplotlib.pyplot as plt


"""
This script creates two CholestericModels based on Cholesterics of different handedness. Then, it creates two 
MixedModels consisting in (1) the right-handed cholesteric on top of the left-handed and (2) the opposite order. The 
reflection spectra of the individual CholestericModels and of these two MixedModels are displayed.
This script is used for our Figure 6.
"""

plt.rcParams.update({'font.size': 12})

# Parameters to choose manually
n_av = 1.433
biref = 0.02
n_e = n_av + 0.5 * biref
n_o = n_av - 0.5 * biref
n_entry = n_av
n_exit = n_av

pitch_nm_1 = 550
pitch_nm_2 = 550
N_hel360_1 = 30
N_hel360_2 = 30

theta_in_deg = -40
theta_in_rad = theta_in_deg * np.pi / 180
wl_min = 400
wl_max = 800
wl_nm_list = np.linspace(start=wl_min, stop=wl_max, num=1 * (wl_max - wl_min), endpoint=False)

refl_chole_1_RCP = []
refl_chole_1_LCP = []
refl_chole_2_RCP = []
refl_chole_2_LCP = []
refl_chole_mix_12_RCP = []
refl_chole_mix_12_LCP = []
refl_chole_mix_21_RCP = []
refl_chole_mix_21_LCP = []

# Cholesteric objects
chole_1 = ch.Cholesteric(pitch360=pitch_nm_1, handedness=1)
chole_2 = ch.Cholesteric(pitch360=pitch_nm_2, handedness=-1)
chole_1_3D = ch.Cholesteric(resolution=12, pitch360=pitch_nm_1, handedness=1)
chole_2_3D = ch.Cholesteric(resolution=12, pitch360=pitch_nm_2, handedness=-1)

# 3D representation of the Cholesterics
fig_ch = plt.figure(constrained_layout=False, figsize=(5.5, 5.5))
gs = fig_ch.add_gridspec(2, 2)
ax_ch1 = fig_ch.add_subplot(gs[0, 0], projection='3d', proj_type='ortho')
ax_ch2 = fig_ch.add_subplot(gs[1, 0], projection='3d', proj_type='ortho')
chole_1_3D.plot_simple_3D(fig_ch, ax_ch1, view="front")
chole_1_3D.plot_add_optics(fig_ch, ax_ch1, theta_in_rad)
ax_ch1.set_yticklabels([])
ax_ch1.set_xticklabels([])
chole_2_3D.plot_simple_3D(fig_ch, ax_ch2, view="front")
chole_2_3D.plot_add_optics(fig_ch, ax_ch2, theta_in_rad)
ax_ch2.set_yticklabels([])
ax_ch2.set_xticklabels([])
#fig_ch.savefig('fig_mixedmodel_ch.png', dpi=300)

# Computation of the reflection spectra
for wl_nm in wl_nm_list:

    # Making the two CholestericModels
    chole_model_1 = ll.CholestericModel(chole_1, n_e, n_o, n_entry, n_exit, wl_nm, N_hel360_1, theta_in_rad)
    chole_model_2 = ll.CholestericModel(chole_2, n_e, n_o, n_entry, n_exit, wl_nm, N_hel360_2, theta_in_rad)

    # Making the two MixedModels
    models_list_12 = [chole_model_1, chole_model_2]
    chole_model_mix_12 = ll.MixedModel(models_list_12, n_entry, n_exit, wl_nm, theta_in_rad)

    models_list_21 = [chole_model_2, chole_model_1]
    chole_model_mix_21 = ll.MixedModel(models_list_21, n_entry, n_exit, wl_nm, theta_in_rad)

    # Reflection of the individual CholestericModels
    refl_1, _ = chole_model_1.get_refl_trans(method="SM", circ=True)
    refl_chole_1_RCP.append(0.5 * (refl_1[0, 0] + refl_1[1, 0]))  # incident RCP
    refl_chole_1_LCP.append(0.5 * (refl_1[1, 1] + refl_1[0, 1]))  # incident LCP
    refl_2, _ = chole_model_2.get_refl_trans(method="SM", circ=True)
    refl_chole_2_RCP.append(0.5 * (refl_2[0, 0] + refl_2[1, 0]))  # incident RCP
    refl_chole_2_LCP.append(0.5 * (refl_2[1, 1] + refl_2[0, 1]))  # incident LCP

    # Reflection of the MixedModels
    refl_mix_SM_12, _ = chole_model_mix_12.get_refl_trans(method="SM", circ=True)
    refl_chole_mix_12_RCP.append((refl_mix_SM_12[0, 0] + refl_mix_SM_12[1, 0]))  # incident RCP
    refl_chole_mix_12_LCP.append((refl_mix_SM_12[1, 1] + refl_mix_SM_12[0, 1]))  # incident LCP
    refl_mix_SM_21, _ = chole_model_mix_21.get_refl_trans(method="SM", circ=True)
    refl_chole_mix_21_RCP.append((refl_mix_SM_21[0, 0] + refl_mix_SM_21[1, 0]))  # incident RCP
    refl_chole_mix_21_LCP.append((refl_mix_SM_21[1, 1] + refl_mix_SM_21[0, 1]))  # incident LCP

    print("Wavelength %s nm done." % (str(wl_nm)))

# Plotting
fig = plt.figure(constrained_layout=True, figsize=(5.2, 3.8))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0, 1])
ax4 = fig.add_subplot(gs[1, 1])

plt.sca(ax1)
plt.plot(wl_nm_list, refl_chole_1_RCP, linestyle='solid', label='RCP')
plt.plot(wl_nm_list, refl_chole_1_LCP, linestyle='solid', label='LCP')
plt.legend()
plt.ylim((-0.1, 1.1))
plt.yticks((0, 1))
plt.xlim([550, 700])
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel('Reflectance')
#plt.grid()

plt.sca(ax2)
plt.plot(wl_nm_list, refl_chole_2_RCP, linestyle='solid', label='RCP')
plt.plot(wl_nm_list, refl_chole_2_LCP, linestyle='solid', label='LCP')
plt.legend()
plt.ylim((-0.1, 1.1))
plt.yticks((0, 1))
plt.xlim([550, 700])
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel('Reflectance')
#plt.grid()

plt.sca(ax3)
plt.plot(wl_nm_list, refl_chole_mix_12_RCP, linestyle='solid', label='RCP')
plt.plot(wl_nm_list, refl_chole_mix_12_LCP, linestyle='solid', label='LCP')
plt.legend()
plt.ylim((-0.1, 1.1))
plt.yticks((0, 1))
plt.xlim([550, 700])
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel('Reflectance')
#plt.grid()

plt.sca(ax4)
plt.plot(wl_nm_list, refl_chole_mix_21_RCP, linestyle='solid', label='RCP')
plt.plot(wl_nm_list, refl_chole_mix_21_LCP, linestyle='solid', label='LCP')
plt.legend()
plt.ylim((-0.1, 1.1))
plt.yticks((0, 1))
plt.xlim([550, 700])
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel('Reflectance')
#plt.grid()

fig.savefig('figure_6_mixedmodel.svg', dpi=300)

plt.show()
