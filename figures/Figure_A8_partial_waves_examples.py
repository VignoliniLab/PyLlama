import numpy as np
import matplotlib.pyplot as plt
import pyllama as ll
import cholesteric as ch
import matplotlib.pylab as pl


"""
This script displays the direction of the partial waves inside a cholesteric liquid crystal and inside the isotropic 
entry half-space.
The outputs from this script are used to construct Figure A8.
"""


# Parameters
theta_in_deg = 20
n_e = 3
n_o = 1.2
n_entry = 2.1
n_exit = 2.1
pitch_nm = 1000
resolution = 360  # therefore: slice number = slice rotation angle in deg
N_per = 1
theta_in_rad = theta_in_deg * np.pi / 180
wl_bragg = 0.5 * (n_e + n_o) * pitch_nm * np.cos(theta_in_rad)

# Create the cholesteric and the cholesteric’s optical model
chole = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=0, N_hel360=1, resolution=resolution)
chole_model = ll.CholestericModel(chole, n_e, n_o, n_entry, n_exit, wl_bragg, N_per, theta_in_rad)
_, _ = chole_model.get_refl_trans()

for klay in [0, 15, 30, 45, 60, 75, 90]:  # with resolution=360, each layer number klay corresponds to the angle of the layer
    print('----')
    print("Rotation angle (deg): " + str(klay))
    angle_layer = int(klay * 360 / resolution)
    angle_layer_rad = klay * 2 * np.pi / resolution
    # Create the figure and the axes. We will generate one figure per layer.
    fig = plt.figure(constrained_layout=False, figsize=(7.5, 2.3))  # (9, 3) for 4 plots  # (12, 3) for 5 plots
    plt.tight_layout()
    ax_sz_sx = fig.add_subplot(151)
    ax_sx_sy = fig.add_subplot(152)
    ax_ex_ey = fig.add_subplot(153)
    ax_hx_hy = fig.add_subplot(154)
    ax_kz_kx = fig.add_subplot(155)
    # Make a database of colours and linestyles
    colours = pl.cm.tab10(np.linspace(0, 1, 10))
    colours_list = [colours[0], colours[0], colours[1], colours[1]]
    linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted']
    p_count = 0
    for p in chole_model.structure.layers[klay].partial_waves:
        # Normalise the values to a custom scale factor. The values are anyway normalised by Python and we’re anyway
        # just interested in the direction of E, H and S so the normalisation helps the visualisation.
        sf = 0.8  # scale factor = the length of the lines on the small plots
        Ex = sf * p.elec[0] / np.sqrt(np.abs(p.elec[0]) ** 2 + abs(p.elec[1]) ** 2)
        Ey = sf * p.elec[1] / np.sqrt(np.abs(p.elec[0]) ** 2 + abs(p.elec[1]) ** 2)
        Hx = sf * p.magnet[0] / np.sqrt(np.abs(p.magnet[0]) ** 2 + abs(p.magnet[1]) ** 2)
        Hy = sf * p.magnet[1] / np.sqrt(np.abs(p.magnet[0]) ** 2 + abs(p.magnet[1]) ** 2)
        Sx = sf * p.poynting[0] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[1]) ** 2)
        Sy = sf * p.poynting[1] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[1]) ** 2)
        # Normalise Poynting twice depending on where I plot it (xy plane or xz plane):
        Sx2 = sf * p.poynting[0] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[2]) ** 2)
        Sz2 = sf * p.poynting[2] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[2]) ** 2)
        # Retrieve k: Kx is in the class Wave, Kz_entry is the eigenvalue. Normalise to 1.
        Kx_plot = p.Kx
        Kz_plot = chole_model.structure.layers[klay].eigenvalues[p_count]
        K_norm = np.sqrt(np.abs(Kx_plot) ** 2 + abs(Kz_plot) ** 2)
        Kx_plot = sf * Kx_plot / K_norm
        Kz_plot = sf * Kz_plot / K_norm
        # Plot the components of the partial wave in their respective plots
        ax_ex_ey.plot([0, Ex], [0, Ey], color=colours_list[p_count], linestyle=linestyle_list[p_count])
        ax_hx_hy.plot([0, Hx], [0, Hy], color=colours_list[p_count], linestyle=linestyle_list[p_count])
        ax_sx_sy.plot([0, Sx], [0, Sy], color=colours_list[p_count], linestyle=linestyle_list[p_count])
        ax_sz_sx.plot([0, Sz2], [0, Sx2], color=colours_list[p_count], linestyle=linestyle_list[p_count])
        ax_kz_kx.plot([0, Kz_plot], [0, Kx_plot], color=colours_list[p_count], linestyle=linestyle_list[p_count])
        p_count = p_count + 1

    # Set up the axes nicely (labels, etc)
    ax_ex_ey.set_xlim((-1, 1))
    ax_ex_ey.set_ylim((-1, 1))
    ax_ex_ey.set_xlabel(r'$E_x$ (a.u.)')
    ax_ex_ey.set_ylabel(r'$E_y$ (a.u.)')
    ax_ex_ey.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
    ax_ex_ey.yaxis.labelpad = -3
    circle1 = plt.Circle((-0.7, -0.7), 0.15, ec='k', fc='w')
    circle1b = plt.Circle((-0.7, -0.7), 0.03, ec=None, fc='k')
    ax_ex_ey.add_artist(circle1)
    ax_ex_ey.add_artist(circle1b)
    ax_ex_ey.set_aspect('equal', adjustable='box')

    ax_hx_hy.set_xlim((-1, 1))
    ax_hx_hy.set_ylim((-1, 1))
    ax_hx_hy.set_xlabel(r'$H_x$ (a.u.)')
    ax_hx_hy.set_ylabel(r'$H_y$ (a.u.)')
    ax_hx_hy.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
    ax_hx_hy.yaxis.labelpad = -3
    circle2 = plt.Circle((-0.7, -0.7), 0.15, ec='k', fc='w')
    circle2b = plt.Circle((-0.7, -0.7), 0.03, ec=None, fc='k')
    ax_hx_hy.add_artist(circle2)
    ax_hx_hy.add_artist(circle2b)
    ax_hx_hy.set_aspect('equal', adjustable='box')

    ax_sx_sy.set_xlim((-1, 1))
    ax_sx_sy.set_ylim((-1, 1))
    ax_sx_sy.set_xlabel(r'$S_x$ (a.u.)')
    ax_sx_sy.set_ylabel(r'$S_y$ (a.u.)')
    ax_sx_sy.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
    ax_sx_sy.yaxis.labelpad = -3
    circle3 = plt.Circle((-0.7, -0.7), 0.15, ec='k', fc='w')
    circle3b = plt.Circle((-0.7, -0.7), 0.03, ec=None, fc='k')
    ax_sx_sy.add_artist(circle3)
    ax_sx_sy.add_artist(circle3b)
    ax_sx_sy.set_aspect('equal', adjustable='box')

    ax_sz_sx.set_xlim((-1, 1))
    ax_sz_sx.set_ylim((-1, 1))
    ax_sz_sx.set_xlabel(r'$S_z$ (a.u.)')
    ax_sz_sx.set_ylabel(r'$S_x$ (a.u.)')
    ax_sz_sx.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
    ax_sz_sx.yaxis.labelpad = -3
    ax_sz_sx.arrow(-0.85, -0.75, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k', head_starts_at_zero=False,
                   length_includes_head=True)
    ax_sz_sx.set_aspect('equal', adjustable='box')

    ax_kz_kx.set_xlim((-1, 1))
    ax_kz_kx.set_ylim((-1, 1))
    ax_kz_kx.set_xlabel(r'$K_z$ (a.u.)')
    ax_kz_kx.set_ylabel(r'$K_x$ (a.u.)')
    ax_kz_kx.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
    ax_kz_kx.yaxis.labelpad = -3
    ax_kz_kx.arrow(-0.85, -0.75, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k', head_starts_at_zero=False,
                   length_includes_head=True)
    ax_kz_kx.set_aspect('equal', adjustable='box')

    # Add a common legend and didte
    fig.legend(['Forward 1', 'Forward 2', 'Backward 1', 'Backward 2'], loc='lower center', ncol=4)
    fig.suptitle("Slice rotation angle = " + str(angle_layer) + "°")
    plt.tight_layout()

    # Save the figure
    figname = "eigenwaves_chole_slice" + str(klay) + ".png"
    #fig.savefig(figname, dpi=300)
    #plt.close('all')

    # Display the figure
    plt.show()


# Same for the entry half-space
fig = plt.figure(constrained_layout=False, figsize=(7.5, 2.3))  # (9, 3) for 4 plots  # (12, 3) for 5 plots
plt.tight_layout()
ax_sz_sx = fig.add_subplot(151)
ax_sx_sy = fig.add_subplot(152)
ax_ex_ey = fig.add_subplot(153)
ax_hx_hy = fig.add_subplot(154)
ax_kz_kx = fig.add_subplot(155)
# Make a database of colours and linestyles
colours = pl.cm.tab10(np.linspace(0, 1, 10))
colours_list = [colours[0], colours[0], colours[1], colours[1]]
linestyle_list = ['solid', 'dashed', 'dashdot', 'dotted']
p_count = 0
for p in chole_model.structure.entry.partial_waves:
    print(p.poynting[0])
    print(p.poynting[1])
    print(p.poynting[2])
    print("----")
    # Normalise the values to a custom scale factor. The values are anyway normalised by Python and we’re anyway
    # just interested in the direction of E, H and S so the normalisation helps the visualisation.
    sf = 0.8  # scale factor = the length of the lines on the small plots
    Ex = sf * p.elec[0] / np.sqrt(np.abs(p.elec[0]) ** 2 + abs(p.elec[1]) ** 2)
    Ey = sf * p.elec[1] / np.sqrt(np.abs(p.elec[0]) ** 2 + abs(p.elec[1]) ** 2)
    Hx = sf * p.magnet[0] / np.sqrt(np.abs(p.magnet[0]) ** 2 + abs(p.magnet[1]) ** 2)
    Hy = sf * p.magnet[1] / np.sqrt(np.abs(p.magnet[0]) ** 2 + abs(p.magnet[1]) ** 2)
    Sx = sf * p.poynting[0] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[1]) ** 2)
    Sy = sf * p.poynting[1] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[1]) ** 2)
    # Normalise Poynting twice depending on where I plot it (xy plane or xz plane):
    Sx2 = sf * p.poynting[0] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[2]) ** 2)
    Sz2 = sf * p.poynting[2] / np.sqrt(np.abs(p.poynting[0]) ** 2 + abs(p.poynting[2]) ** 2)
    # Retrieve k: Kx is in the class Wave, Kz_entry is the eigenvalue. Normalise to 1.
    Kx_plot = p.Kx
    Kz_plot = chole_model.structure.layers[klay].eigenvalues[p_count]
    K_norm = np.sqrt(np.abs(Kx_plot) ** 2 + abs(Kz_plot) ** 2)
    Kx_plot = sf * Kx_plot / K_norm
    Kz_plot = sf * Kz_plot / K_norm
    # Plot the components of the partial wave in their respective plots
    ax_ex_ey.plot([0, Ex], [0, Ey], color=colours_list[p_count], linestyle=linestyle_list[p_count])
    ax_hx_hy.plot([0, Hx], [0, Hy], color=colours_list[p_count], linestyle=linestyle_list[p_count])
    ax_sx_sy.plot([0, Sx], [0, Sy], color=colours_list[p_count], linestyle=linestyle_list[p_count])
    ax_sz_sx.plot([0, Sz2], [0, Sx2], color=colours_list[p_count], linestyle=linestyle_list[p_count])
    ax_kz_kx.plot([0, Kz_plot], [0, Kx_plot], color=colours_list[p_count], linestyle=linestyle_list[p_count])
    p_count = p_count + 1

# Set up the axes nicely (labels, etc)
ax_ex_ey.set_xlim((-1, 1))
ax_ex_ey.set_ylim((-1, 1))
ax_ex_ey.set_xlabel(r'$E_x$ (a.u.)')
ax_ex_ey.set_ylabel(r'$E_y$ (a.u.)')
ax_ex_ey.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
ax_ex_ey.yaxis.labelpad = -3
circle1 = plt.Circle((-0.7, -0.7), 0.15, ec='k', fc='w')
circle1b = plt.Circle((-0.7, -0.7), 0.03, ec=None, fc='k')
ax_ex_ey.add_artist(circle1)
ax_ex_ey.add_artist(circle1b)
# ax_ex_ey.plot([0, np.cos(angle_layer_rad)], [0, np.sin(angle_layer_rad)], color='k', linestyle='dotted')
ax_ex_ey.set_aspect('equal', adjustable='box')

ax_hx_hy.set_xlim((-1, 1))
ax_hx_hy.set_ylim((-1, 1))
ax_hx_hy.set_xlabel(r'$H_x$ (a.u.)')
ax_hx_hy.set_ylabel(r'$H_y$ (a.u.)')
ax_hx_hy.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
ax_hx_hy.yaxis.labelpad = -3
circle2 = plt.Circle((-0.7, -0.7), 0.15, ec='k', fc='w')
circle2b = plt.Circle((-0.7, -0.7), 0.03, ec=None, fc='k')
ax_hx_hy.add_artist(circle2)
ax_hx_hy.add_artist(circle2b)
ax_hx_hy.set_aspect('equal', adjustable='box')

ax_sx_sy.set_xlim((-1, 1))
ax_sx_sy.set_ylim((-1, 1))
ax_sx_sy.set_xlabel(r'$S_x$ (a.u.)')
ax_sx_sy.set_ylabel(r'$S_y$ (a.u.)')
ax_sx_sy.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
ax_sx_sy.yaxis.labelpad = -3
circle3 = plt.Circle((-0.7, -0.7), 0.15, ec='k', fc='w')
circle3b = plt.Circle((-0.7, -0.7), 0.03, ec=None, fc='k')
ax_sx_sy.add_artist(circle3)
ax_sx_sy.add_artist(circle3b)
ax_sx_sy.set_aspect('equal', adjustable='box')

ax_sz_sx.set_xlim((-1, 1))
ax_sz_sx.set_ylim((-1, 1))
ax_sz_sx.set_xlabel(r'$S_z$ (a.u.)')
ax_sz_sx.set_ylabel(r'$S_x$ (a.u.)')
ax_sz_sx.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
ax_sz_sx.yaxis.labelpad = -3
ax_sz_sx.arrow(-0.85, -0.75, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k', head_starts_at_zero=False,
               length_includes_head=True)
ax_sz_sx.set_aspect('equal', adjustable='box')

ax_kz_kx.set_xlim((-1, 1))
ax_kz_kx.set_ylim((-1, 1))
ax_kz_kx.set_xlabel(r'$K_z$ (a.u.)')
ax_kz_kx.set_ylabel(r'$K_x$ (a.u.)')
ax_kz_kx.xaxis.labelpad = 0  # write Ex and Ey closer to the axes
ax_kz_kx.yaxis.labelpad = -3
ax_kz_kx.arrow(-0.85, -0.75, 0.4, 0, head_width=0.1, head_length=0.1, fc='k', ec='k', head_starts_at_zero=False,
               length_includes_head=True)
ax_kz_kx.set_aspect('equal', adjustable='box')

# Add a common legend and didte
fig.legend(['Forward 1', 'Forward 2', 'Backward 1', 'Backward 2'], loc='lower center', ncol=4)
fig.suptitle("Entry half-space")
plt.tight_layout()

# Save the figure
figname = "eigenwaves_chole_entry.png"
#fig.savefig(figname, dpi=300)
#plt.close('all')

# Display the figure
plt.show()


