import numpy as np
import matplotlib.pyplot as plt
import pyllama as sc
import cholesteric as ch
import matplotlib.pylab as pl


"""
This script plots the partial waves from a cholesteric slice and compares with Oldano's formulas for uniaxial crystals.
Outputs from this script are used for Figure 4.
"""


# De Vries says that the direction of Ex and Ey will follow the slice's rotation angle

# Display parameter
np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

theta_in_deg = 60  #60  #45  #0 # 20
n_av = 1.433
biref = 0.035
n_e = n_av + 0.5 * biref
n_o = n_av - 0.5 * biref
n_entry = 0.5 * (n_e + n_o)
N_hel360 = 1
pitch_nm = 500
theta_in_rad = theta_in_deg * np.pi / 180
wl_bragg = 0.5 * (n_e + n_o) * pitch_nm * np.cos(theta_in_rad)

# Calculation with Scatmat
n_exit = n_entry
thickness_nm = 500
eps0 = np.array([[n_e ** 2, 0, 0],
                 [0, n_o ** 2, 0],
                 [0, 0, n_o ** 2]])

resolution = 360
chole = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=0, N_hel360=1, resolution=resolution)

model_chole = sc.CholestericModel(chole, n_e, n_o, n_entry, n_exit, wl_bragg, N_hel360, theta_in_rad)
model_to_plot = model_chole

fig = plt.figure(constrained_layout=False, figsize=(7.2, 2.1)) # figsize=(7.5, 2.3))  # (9, 3) for 4 plots  # (12, 3) for 5 plots
plt.tight_layout()
ax1 = fig.add_subplot(151)
ax2 = fig.add_subplot(152)
ax3 = fig.add_subplot(153)
ax4 = fig.add_subplot(154)
ax5 = fig.add_subplot(155)
ax_list = [ax1, ax2, ax3, ax4, ax5]

lay_list = [0, 15, 30, 45, 60]
for lay in range(0, len(lay_list), 1):
    klay = lay_list[lay]
    ax = ax_list[lay]
    angle_layer = int(klay * 360 / resolution)
    angle_layer_rad = klay * 2 * np.pi / resolution
    angle_eta_rad = angle_layer_rad + np.pi / 2
    colours = pl.cm.tab10(np.linspace(0, 1, 10))
    colours_list = [colours[0], colours[0], colours[1], colours[1]]
    linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted']  # (0, (8, 10)) instead of dashdot
    # Calculation of Ex and Ey with Oldano’s formulas (DOI 10.1103/physreva.40.6014, we use the same notations)
    # Parameters for the calculations:
    oldano_theta = np.pi / 2  #theta_in_rad  # polar (corresponds to PyLlama’s angle around y)
    oldano_phi = angle_layer_rad  # polar (corresponds to PyLlama’s angle around z)
    epso = n_o ** 2
    epse = n_e ** 2
    epst = 1 / (((np.cos(oldano_theta) ** 2) / epso) + ((np.sin(oldano_theta) ** 2) / epse))
    epsf = epso * np.sin(oldano_phi) ** 2 + epst * np.cos(oldano_phi) ** 2
    m = n_entry * np.sin(theta_in_rad)
    mt = m * (1 - epst / epso) * (np.cos(oldano_theta) / np.sin(oldano_theta))
    de_1 = mt * np.cos(oldano_phi) + np.sqrt(epst - m ** 2 * (epst * epsf) / (epse * epso))
    de_2 = mt * np.cos(oldano_phi) - np.sqrt(epst - m ** 2 * (epst * epsf) / (epse * epso))
    do_1 = np.sqrt(epso - m ** 2)
    do_2 = - np.sqrt(epso - m ** 2)
    # Eigenvectors from Oldano: (and we normalise them)
    pe_1 = np.array(
        [[(1 - m ** 2 / epso) * np.cos(oldano_phi) - m * np.cos(oldano_theta) * de_1 / (epso * np.sin(oldano_theta))],
         [de_1 * np.cos(oldano_phi) - m * np.cos(oldano_theta) / np.sin(oldano_theta)],
         [np.sin(oldano_phi)],
         [de_1 * np.sin(oldano_phi)]])
    norm_e_1 = np.linalg.norm(pe_1)
    pe_1 = pe_1 / norm_e_1

    pe_2 = np.array(
        [[(1 - m ** 2 / epso) * np.cos(oldano_phi) - m * np.cos(oldano_theta) * de_2 / (epso * np.sin(oldano_theta))],
         [de_2 * np.cos(oldano_phi) - m * np.cos(oldano_theta) / np.sin(oldano_theta)],
         [np.sin(oldano_phi)],
         [de_2 * np.sin(oldano_phi)]])
    norm_e_2 = np.linalg.norm(pe_2)
    pe_2 = pe_2 / norm_e_2

    po_1 = np.array([[-np.sin(oldano_phi)],
                     [- epso * np.sin(oldano_phi) / do_1],
                     [np.cos(oldano_phi) - m * np.cos(oldano_theta) / (do_1 * np.sin(oldano_theta))],
                     [do_1 * np.cos(oldano_phi) - m * np.cos(oldano_theta) / np.sin(oldano_theta)]])
    norm_o_1 = np.linalg.norm(po_1)
    po_1 = po_1 / norm_o_1

    po_2 = np.array([[-np.sin(oldano_phi)],
                     [- epso * np.sin(oldano_phi) / do_2],
                     [np.cos(oldano_phi) - m * np.cos(oldano_theta) / (do_2 * np.sin(oldano_theta))],
                     [do_2 * np.cos(oldano_phi) - m * np.cos(oldano_theta) / np.sin(oldano_theta)]])
    norm_o_2 = np.linalg.norm(po_2)
    po_2 = po_2 / norm_o_2
    # Plotting of the eigenvectors from Oldano
    ax.plot([- 5 * pe_1[0], 5 * pe_1[0]], [-5 * pe_1[2], 5 * pe_1[2]], color=[0.8, 0.8, 0.8], linestyle='solid',
            linewidth=4)
    ax.plot([- 5 * po_1[0], 5 * po_1[0]], [-5 * po_1[2], 5 * po_1[2]], color=[0.8, 0.8, 0.8], linestyle='solid',
            linewidth=4, label='_nolegend_')

    # Plotting of the partial waves from PyLlama
    p_count = 0
    for p in model_to_plot.structure.layers[klay].partial_waves:
        sf = 0.8  # scale factor
        Ex = sf * p.elec[0] / np.sqrt(np.abs(p.elec[0]) ** 2 + abs(p.elec[1]) ** 2)
        Ey = sf * p.elec[1] / np.sqrt(np.abs(p.elec[0]) ** 2 + abs(p.elec[1]) ** 2)
        # Plot
        ax.plot([0, Ex], [0, Ey], color=colours_list[p_count], linestyle=linestyle_list[p_count])
        p_count = p_count + 1

    # General setting-up of the figure
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    # ax.legend(['Forward 1', 'Forward 2', 'Backward 1', 'Backward 2'])
    ax.set_xlabel(r'$E_x$ (a.u.)')
    ax.set_ylabel(r'$E_y$ (a.u.)')
    ax.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
    ax.yaxis.labelpad = -3
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(str(angle_layer) + "°")

fig.legend(['Analytical', 'Forward 1', 'Forward 2', 'Backward 1', 'Backward 2'], loc='lower center',
           ncol=5, framealpha=1, prop={'size': 9})
#fig_sorting_waves.subplots_adjust(top=0.5)

plt.tight_layout()

#fig.savefig("figure_4_e_f_eigenvectors", dpi=300)

plt.show()





