import numpy as np
import matplotlib.pyplot as plt
import pyllama as ll
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec


"""
This script reproduces the situation described in Rivière, DOI 10.1016/0030-4018(78)90308-5, and highlights the 
necessity of sorting the partial waves according to the Poynting vector.
In PyLlama, the partial waves are sorted in the function Layer._sort_p_q(). If the user wishes to see what happens 
when they are sorted according to the z component of the wavevector, the user should edit PyLlama by:
    - un-commenting the paragraph after "# Analysing with Kz (sometimes fails):" in Layer._sort_p_q()
    - commenting the paragraph after "# Analyse with the Poynting vector:" in Layer._sort_p_q()
Sorting with the wavevector is not a normal behaviour of PyLlama and no option is available to the user.
The output of this script (with the sorting according to the wavevector) corresponds to Figure A9.
When the partial waves are sorted differently (Poynting vector, wavevector) or calculated differently (Numpy, Sympy), 
the user should expect a sign change between pairs of partial waves.
"""


# Display parameter
plt.rcParams.update({'font.size': 10})
np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})
colours = pl.cm.tab10(np.linspace(0, 1, 10))
colours_list = [colours[1], colours[0], colours[3], colours[2], colours[6], colours[9]]
linestyle_list = ['solid', 'solid', (0, (4, 4)), (0, (4, 4))]

# System parameters
n_av = 1.433
biref = 0.5
n_e = n_av + 0.5 * biref
n_o = n_av - 0.5 * biref
n_entry = 1.9
n_exit = n_av

wl_nm = 500

thickness_nm = 4000
eps0 = np.array([[n_e ** 2, 0, 0],
                 [0, n_o ** 2, 0],
                 [0, 0, n_o ** 2]])
rotangle_deg = 45
rotangle_rad = rotangle_deg * np.pi / 180
rotaxis = 'y'
eps_rot = ll.Layer.rotate_permittivity(eps0, rotangle_rad, 'y')

# Critical angles
sin_critical_angle_rad = (1 / n_entry) * np.sqrt((n_o * np.cos(rotangle_rad)) ** 2 + (n_e * np.sin(rotangle_rad)) ** 2)
critical_angle_rad = np.arcsin(sin_critical_angle_rad)
critical_angle_deg_p = critical_angle_rad * 180 / np.pi
critical_angle_o_rad = np.arcsin(n_o / n_entry)
critical_angle_deg_s = critical_angle_o_rad * 180 / np.pi

# Empty variables
theta_in_deg_list = np.arange(0, 90, 0.01)
refl_p = []
refl_s = []
list_of_partial_waves = []
list_of_eigenvalues = []
Kz_works = 0
Kz_does_not_work = 0


# Calculation
def get_data(theta_in_deg):
    theta_in_rad = theta_in_deg * np.pi / 180

    # Make the model
    model = ll.SlabModel(eps0, thickness_nm, n_entry, n_exit, wl_nm, theta_in_rad, rotangle_rad, rotaxis)

    # Calculate the reflectance and store it
    refl, trans = model.get_refl_trans(method="TM", circ=False)
    refl_p.append(refl[0, 0])
    refl_s.append(refl[1, 1])

    partial_waves = model.structure.layers[0].partial_waves
    eigenvalues = model.structure.layers[0].eigenvalues

    return partial_waves, eigenvalues

kt = -1
for theta_in_deg in theta_in_deg_list:
    kt += 1
    partial_waves, eigenvalues = get_data(theta_in_deg)
    list_of_partial_waves.append(partial_waves)
    list_of_eigenvalues.append(eigenvalues)

    # Analyse if sorting with Kz works
    id_refl = []
    id_trans = []
    for kp in range(4):
        current_Kz = eigenvalues[kp]
        if np.isreal(current_Kz):
            if current_Kz >= 0:
                id_trans.append(kp)
            else:
                id_refl.append(kp)
        else:
            if np.imag(current_Kz) >= 0:
                id_trans.append(kp)
            else:
                id_refl.append(kp)
    if (len(id_refl) == 2 and len(id_trans) == 2):
        if Kz_does_not_work == 0:
            Kz_works = kt
    else:
        Kz_does_not_work = kt
    print(theta_in_deg)

# Plotting
fig_S_K = plt.figure(constrained_layout=False, figsize=(9, 4))
gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig_S_K)
ax_refl = fig_S_K.add_subplot(gs[1, 0])
ax_Sz_real = fig_S_K.add_subplot(gs[0, 1])
ax_Sz_imag = fig_S_K.add_subplot(gs[1, 1])
ax_Kz_real = fig_S_K.add_subplot(gs[0, 2])
ax_Kz_imag = fig_S_K.add_subplot(gs[1, 2])

plt.sca(ax_refl)
ax_refl.plot(theta_in_deg_list, refl_s, color=colours_list[4], linewidth=1, label="s")
ax_refl.plot(theta_in_deg_list, refl_p, color=colours_list[5], linewidth=1, label="p")
ax_refl.plot([critical_angle_deg_s, critical_angle_deg_s], [-0.1, 1.1],
             linestyle='dashdot',
             color='k',
             linewidth=1.5,
             label=r'$\theta_s$')
ax_refl.plot([critical_angle_deg_p, critical_angle_deg_p], [-0.1, 1.1],
             linestyle='dotted',
             color='k',
             linewidth=1.5,
             label=r'$\theta_p$')
plt.xlabel(r'$\theta_{in}$ (deg)')
ax_refl.set_ylabel('Reflectance')
ax_refl.set_xlim((0, 90))
ax_refl.set_ylim((-0.1, 1.1))
plt.legend()

for kp in range(4):
    current_label = str(kp)
    poynting_x = []
    poynting_y = []
    poynting_z = []
    k_z = []
    kt = -1
    for theta_in_deg in theta_in_deg_list:
        kt += 1
        current_partial_wave = list_of_partial_waves[kt][kp]
        poynting_x.append(current_partial_wave.poynting[0])
        poynting_y.append(current_partial_wave.poynting[1])
        poynting_z.append(current_partial_wave.poynting[2])
        current_eigenvalue = list_of_eigenvalues[kt][kp]
        k_z.append(current_eigenvalue)

    plt.sca(ax_Sz_real)
    plt.plot(theta_in_deg_list, np.real(poynting_z),
             color=colours_list[kp],
             linestyle=linestyle_list[kp],
             linewidth=2)
    plt.ylabel(r'Re($S_z$)')

    plt.sca(ax_Sz_imag)
    plt.plot(theta_in_deg_list, np.imag(poynting_z),
             color=colours_list[kp],
             linestyle=linestyle_list[kp],
             linewidth=2)
    plt.xlabel(r'$\theta_{in}$ (deg)')
    plt.ylabel(r'Im($S_z$)')

    plt.sca(ax_Kz_real)
    plt.plot(theta_in_deg_list, np.real(k_z),
             color=colours_list[kp],
             linestyle=linestyle_list[kp],
             linewidth=2)
    plt.ylabel(r'Re($K_z$)')

    plt.sca(ax_Kz_imag)
    plt.plot(theta_in_deg_list, np.imag(k_z),
             color=colours_list[kp],
             linestyle=linestyle_list[kp],
             linewidth=2)
    plt.xlabel(r'$\theta_{in}$ (deg)')
    plt.ylabel(r'Im($K_z$)')

print("Critical angles:")
print(critical_angle_deg_s)
print(critical_angle_deg_p)

for ax in [ax_Sz_real, ax_Sz_imag, ax_Kz_real, ax_Kz_imag]:
    plt.sca(ax)
    ax.set_xlim((0, 90))
    ax.set_ylim((-1.7, 1.7))
    plt.yticks(np.arange(-1, 1.05, step=1))
    ax.yaxis.labelpad = -5  # bring the label closer to the tick labels
    plt.plot([critical_angle_deg_s, critical_angle_deg_s], [-2, 2],
             color="k", linestyle="dashdot", label=r'$\theta_s$', linewidth=1.5)
    plt.plot([critical_angle_deg_p, critical_angle_deg_p], [-2, 2],
             color="k", linestyle="dotted", label=r'$\theta_p$', linewidth=1.5)

for ax in [ax_refl, ax_Sz_real, ax_Sz_imag, ax_Kz_real, ax_Kz_imag]:
    plt.sca(ax)
    rect = plt.Rectangle((theta_in_deg_list[Kz_works], -2),
                         (theta_in_deg_list[Kz_does_not_work] - theta_in_deg_list[Kz_works]), 4,
                         facecolor=(0, 0, 0, 0.2))
    ax.add_patch(rect)
    plt.grid(linestyle='dotted', axis="x")
    plt.grid(linestyle='dotted', axis="y")

plt.sca(ax_Sz_real)
plt.legend()

ax_Sz_real.axes.xaxis.set_ticklabels([])
ax_Kz_real.axes.xaxis.set_ticklabels([])

plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0.35)

fig_S_K.savefig("figure_A9_sorting_partial_waves.png", dpi=300)

plt.show()
