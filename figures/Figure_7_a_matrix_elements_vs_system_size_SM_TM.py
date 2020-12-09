import pyllama as ll
import numpy as np
import cholesteric as ch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle


"""
This scripts records:
    - the elements in the transfer matrix (TM)
    - the elements in the scattering matrix (SM)
    - the reflection coefficients calculated with the TM
    - the reflection coefficients calculated with the SM
from a system of increasing size. The base system is a cholesteric domain with 125 pitches. Sub-systems of incremental 
sizes are extracted from the cholesteric, and the matrices are calculated from these sub-systems. The user should note 
that all sub-systems are embedded inside entry and exit isotropic media.
The objective of this code is to visualise the divergence of the elements in the TM when the system becomes 
bigger, while the elements in the SM don’t diverge.
The output from this script is used on Figure 7a. This script shows all the matrix elements while only a subset was 
used on Figure 7a.
"""


def plot_one_block(ax, idx, titlestring, data):
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0
    plt.axes(ax)
    plt.scatter(data[idx, 0, idx_wl, :stop:step], data[idx, 1, idx_wl, :stop:step], c=colourlist, cmap=common_cmap, s=10)
    plt.axis(axislim)
    plt.xlabel('real')
    plt.ylabel('imag')
    plt.title(titlestring)
    ax.ticklabel_format(style='sci', scilimits=(0, 2))


def plot_all_blocks(data):
    plot_one_block(ax_00, 0, "M[0, 0]", data)
    plot_one_block(ax_01, 1, "M[0, 1]", data)
    plot_one_block(ax_10, 2, "M[1, 0]", data)
    plot_one_block(ax_11, 3, "M[1, 1]", data)
    plot_one_block(ax_02, 4, "M[0, 2]", data)
    plot_one_block(ax_03, 5, "M[0, 3]", data)
    plot_one_block(ax_12, 6, "M[1, 2]", data)
    plot_one_block(ax_13, 7, "M[1, 3]", data)
    plot_one_block(ax_20, 8, "M[2, 0]", data)
    plot_one_block(ax_21, 9, "M[2, 1]", data)
    plot_one_block(ax_30, 10, "M[3, 0]", data)
    plot_one_block(ax_31, 11, "M[3, 1]", data)
    plot_one_block(ax_22, 12, "M[2, 2]", data)
    plot_one_block(ax_23, 13, "M[2, 3]", data)
    plot_one_block(ax_32, 14, "M[3, 2]", data)
    plot_one_block(ax_33, 15, "M[3, 3]", data)


# Parameters for the cholesteric
n_av = 1.433
biref = 0.09
n_e = n_av + 0.5 * biref
n_o = n_av - 0.5 * biref
n_entry = n_av
n_exit = n_av
pitch_nm = 500
N_hel360 = 125
theta_in_deg = 0
theta_in_rad = theta_in_deg * np.pi / 180
wl_bragg = n_av * pitch_nm * np.cos(theta_in_rad)
wl_min = 400
wl_max = 800
wl_nm_list = [wl_bragg]
reso = 20
lay_list = list(range(1, int(N_hel360 * reso + 1), 1))
un = 1

chole = ch.Cholesteric(pitch360=pitch_nm, resolution=reso, N_hel360=N_hel360, handedness=1)

# Initialisation of empty arrays for results
data_reflco_SM = np.empty((4, 2, len(wl_nm_list), len(lay_list)))
data_reflco_SM[:] = np.NaN

data_reflco_TM = np.empty((4, 2, len(wl_nm_list), len(lay_list)))
data_reflco_TM[:] = np.NaN

data_SM = np.empty((16, 2, len(wl_nm_list), len(lay_list)))
data_SM[:] = np.NaN

data_TM = np.empty((16, 2, len(wl_nm_list), len(lay_list)))
data_TM[:] = np.NaN

# Calculation
kl = 0
for wl in wl_nm_list:
    chole_model = ll.CholestericModel(chole, n_e, n_o, n_entry, n_exit, wl, un, theta_in_rad)
    stack_model = ll.Model.copy_as_stack(chole_model)
    klay = 0
    for lay in lay_list:
        sub_stack = stack_model.extract_stack(0, lay)

        res_reflco_SM, res_transco_SM = sub_stack.structure._get_fresnel_SM()
        res_reflco_TM, res_transco_TM = sub_stack.structure._get_fresnel_TM()
        res_reflco_SM, _ = ll.Structure.fresnel_to_fresnel_circ(res_reflco_SM, res_transco_SM)
        res_reflco_TM, _ = ll.Structure.fresnel_to_fresnel_circ(res_reflco_TM, res_transco_TM)

        res_SM = sub_stack.structure.build_scattering_matrix()
        res_TM = sub_stack.structure.build_transfer_matrix()

        # Storage of the results
        data_reflco_SM[0, 0, kl, klay] = np.real(res_reflco_SM[0, 0])
        data_reflco_SM[1, 0, kl, klay] = np.real(res_reflco_SM[0, 1])
        data_reflco_SM[2, 0, kl, klay] = np.real(res_reflco_SM[1, 0])
        data_reflco_SM[3, 0, kl, klay] = np.real(res_reflco_SM[1, 1])

        data_reflco_SM[0, 1, kl, klay] = np.imag(res_reflco_SM[0, 0])
        data_reflco_SM[1, 1, kl, klay] = np.imag(res_reflco_SM[0, 1])
        data_reflco_SM[2, 1, kl, klay] = np.imag(res_reflco_SM[1, 0])
        data_reflco_SM[3, 1, kl, klay] = np.imag(res_reflco_SM[1, 1])

        data_reflco_TM[0, 0, kl, klay] = np.real(res_reflco_TM[0, 0])
        data_reflco_TM[1, 0, kl, klay] = np.real(res_reflco_TM[0, 1])
        data_reflco_TM[2, 0, kl, klay] = np.real(res_reflco_TM[1, 0])
        data_reflco_TM[3, 0, kl, klay] = np.real(res_reflco_TM[1, 1])

        data_reflco_TM[0, 1, kl, klay] = np.imag(res_reflco_TM[0, 0])
        data_reflco_TM[1, 1, kl, klay] = np.imag(res_reflco_TM[0, 1])
        data_reflco_TM[2, 1, kl, klay] = np.imag(res_reflco_TM[1, 0])
        data_reflco_TM[3, 1, kl, klay] = np.imag(res_reflco_TM[1, 1])

        data_SM[0, 0, kl, klay] = np.real(res_SM[0, 0])
        data_SM[1, 0, kl, klay] = np.real(res_SM[0, 1])
        data_SM[2, 0, kl, klay] = np.real(res_SM[1, 0])
        data_SM[3, 0, kl, klay] = np.real(res_SM[1, 1])
        data_SM[4, 0, kl, klay] = np.real(res_SM[0, 2])
        data_SM[5, 0, kl, klay] = np.real(res_SM[0, 3])
        data_SM[6, 0, kl, klay] = np.real(res_SM[1, 2])
        data_SM[7, 0, kl, klay] = np.real(res_SM[1, 3])
        data_SM[8, 0, kl, klay] = np.real(res_SM[2, 0])
        data_SM[9, 0, kl, klay] = np.real(res_SM[2, 1])
        data_SM[10, 0, kl, klay] = np.real(res_SM[3, 0])
        data_SM[11, 0, kl, klay] = np.real(res_SM[3, 1])
        data_SM[12, 0, kl, klay] = np.real(res_SM[2, 2])
        data_SM[13, 0, kl, klay] = np.real(res_SM[2, 3])
        data_SM[14, 0, kl, klay] = np.real(res_SM[3, 2])
        data_SM[15, 0, kl, klay] = np.real(res_SM[3, 3])

        data_SM[0, 1, kl, klay] = np.imag(res_SM[0, 0])
        data_SM[1, 1, kl, klay] = np.imag(res_SM[0, 1])
        data_SM[2, 1, kl, klay] = np.imag(res_SM[1, 0])
        data_SM[3, 1, kl, klay] = np.imag(res_SM[1, 1])
        data_SM[4, 1, kl, klay] = np.imag(res_SM[0, 2])
        data_SM[5, 1, kl, klay] = np.imag(res_SM[0, 3])
        data_SM[6, 1, kl, klay] = np.imag(res_SM[1, 2])
        data_SM[7, 1, kl, klay] = np.imag(res_SM[1, 3])
        data_SM[8, 1, kl, klay] = np.imag(res_SM[2, 0])
        data_SM[9, 1, kl, klay] = np.imag(res_SM[2, 1])
        data_SM[10, 1, kl, klay] = np.imag(res_SM[3, 0])
        data_SM[11, 1, kl, klay] = np.imag(res_SM[3, 1])
        data_SM[12, 1, kl, klay] = np.imag(res_SM[2, 2])
        data_SM[13, 1, kl, klay] = np.imag(res_SM[2, 3])
        data_SM[14, 1, kl, klay] = np.imag(res_SM[3, 2])
        data_SM[15, 1, kl, klay] = np.imag(res_SM[3, 3])

        data_TM[0, 0, kl, klay] = np.real(res_TM[0, 0])
        data_TM[1, 0, kl, klay] = np.real(res_TM[0, 1])
        data_TM[2, 0, kl, klay] = np.real(res_TM[1, 0])
        data_TM[3, 0, kl, klay] = np.real(res_TM[1, 1])
        data_TM[4, 0, kl, klay] = np.real(res_TM[0, 2])
        data_TM[5, 0, kl, klay] = np.real(res_TM[0, 3])
        data_TM[6, 0, kl, klay] = np.real(res_TM[1, 2])
        data_TM[7, 0, kl, klay] = np.real(res_TM[1, 3])
        data_TM[8, 0, kl, klay] = np.real(res_TM[2, 0])
        data_TM[9, 0, kl, klay] = np.real(res_TM[2, 1])
        data_TM[10, 0, kl, klay] = np.real(res_TM[3, 0])
        data_TM[11, 0, kl, klay] = np.real(res_TM[3, 1])
        data_TM[12, 0, kl, klay] = np.real(res_TM[2, 2])
        data_TM[13, 0, kl, klay] = np.real(res_TM[2, 3])
        data_TM[14, 0, kl, klay] = np.real(res_TM[3, 2])
        data_TM[15, 0, kl, klay] = np.real(res_TM[3, 3])

        data_TM[0, 1, kl, klay] = np.imag(res_TM[0, 0])
        data_TM[1, 1, kl, klay] = np.imag(res_TM[0, 1])
        data_TM[2, 1, kl, klay] = np.imag(res_TM[1, 0])
        data_TM[3, 1, kl, klay] = np.imag(res_TM[1, 1])
        data_TM[4, 1, kl, klay] = np.imag(res_TM[0, 2])
        data_TM[5, 1, kl, klay] = np.imag(res_TM[0, 3])
        data_TM[6, 1, kl, klay] = np.imag(res_TM[1, 2])
        data_TM[7, 1, kl, klay] = np.imag(res_TM[1, 3])
        data_TM[8, 1, kl, klay] = np.imag(res_TM[2, 0])
        data_TM[9, 1, kl, klay] = np.imag(res_TM[2, 1])
        data_TM[10, 1, kl, klay] = np.imag(res_TM[3, 0])
        data_TM[11, 1, kl, klay] = np.imag(res_TM[3, 1])
        data_TM[12, 1, kl, klay] = np.imag(res_TM[2, 2])
        data_TM[13, 1, kl, klay] = np.imag(res_TM[2, 3])
        data_TM[14, 1, kl, klay] = np.imag(res_TM[3, 2])
        data_TM[15, 1, kl, klay] = np.imag(res_TM[3, 3])

        print(lay)
        klay = klay + 1
    kl = kl + 1
    print(wl)


dict_to_save = dict(data_reflco_SM=data_reflco_SM, data_reflco_TM=data_reflco_TM, data_SM=data_SM, data_TM=data_TM,
                    wl_nm_list=wl_nm_list, n_av=n_av, biref=biref, n_entry=n_entry,
                    n_exit=n_exit, reso=reso, lay_list=lay_list, pitch_nm=pitch_nm, theta_in_deg=theta_in_deg,
                    chole=chole, wl_bragg=wl_bragg, N_hel360=N_hel360)

# Save
filetitle = "output.pck"
with open(filetitle, 'wb') as f:
    pickle.dump(dict_to_save, f)

# Open and plot
filetitle = "output.pck"
f = open(filetitle, 'rb')
d = pickle.load(f)
data_reflco_SM = d['data_reflco_SM']
data_reflco_TM = d['data_reflco_TM']
data_SM = d['data_SM']
data_TM = d['data_TM']
wl_nm_list = d['wl_nm_list']
n_av = d['n_av']
biref = d['biref']
n_entry = d['n_entry']
n_exit = d['n_exit']
reso = d['reso']
lay_list = d['lay_list']
pitch_nm = d['pitch_nm']
theta_in_deg = d['theta_in_deg']
chole = d['chole']
wl_bragg = d["wl_bragg"]
N_hel360 = d["N_hel360"]


lay_list_normalised = [lay / reso for lay in lay_list]
label_list = ('RCP to RCP', 'LCP to RCP', 'RCP to LCP', 'LCP to LCP')
idx_wl = 0
xlimlist = (0, max(lay_list_normalised))

fig = plt.figure(constrained_layout=False, figsize=(9, 5))
plt.tight_layout()
widths = [1, 1, 1, 1]
heights = [1, 1]
gs = fig.add_gridspec(2, 4, width_ratios=widths, height_ratios=heights)
ax00 = fig.add_subplot(gs[0, 0])
ax01 = fig.add_subplot(gs[0, 1])
ax02 = fig.add_subplot(gs[0, 2])
ax03 = fig.add_subplot(gs[0, 3])
ax10 = fig.add_subplot(gs[1, 0])
ax11 = fig.add_subplot(gs[1, 1])
ax12 = fig.add_subplot(gs[1, 2])
ax13 = fig.add_subplot(gs[1, 3])

ax00.plot(lay_list_normalised, data_reflco_SM[0, 0, idx_wl, :])
ax00.plot(lay_list_normalised, data_reflco_SM[1, 0, idx_wl, :])
ax00.plot(lay_list_normalised, data_reflco_SM[2, 0, idx_wl, :], linestyle='dashed')
ax00.plot(lay_list_normalised, data_reflco_SM[3, 0, idx_wl, :])
ax00.set_title(r'$r_{SM}$, real')
ax00.set_ylim((-1, 1))
plt.axes(ax00)
plt.legend(label_list)
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax01.plot(lay_list_normalised, data_reflco_SM[0, 1, idx_wl, :])
ax01.plot(lay_list_normalised, data_reflco_SM[1, 1, idx_wl, :])
ax01.plot(lay_list_normalised, data_reflco_SM[2, 1, idx_wl, :], linestyle='dashed')
ax01.plot(lay_list_normalised, data_reflco_SM[3, 1, idx_wl, :])
ax01.set_title(r'$r_{SM}$, imag')
ax01.set_ylim((-1, 1))
plt.axes(ax01)
plt.legend(label_list)
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax10.plot(lay_list_normalised, data_reflco_TM[0, 0, idx_wl, :])
ax10.plot(lay_list_normalised, data_reflco_TM[1, 0, idx_wl, :])
ax10.plot(lay_list_normalised, data_reflco_TM[2, 0, idx_wl, :], linestyle='dashed')
ax10.plot(lay_list_normalised, data_reflco_TM[3, 0, idx_wl, :])
ax10.set_title(r'$r_{TM}$, real')
ax10.set_ylim((-1, 1))
plt.axes(ax10)
plt.legend(label_list)
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax11.plot(lay_list_normalised, data_reflco_TM[0, 1, idx_wl, :])
ax11.plot(lay_list_normalised, data_reflco_TM[1, 1, idx_wl, :])
ax11.plot(lay_list_normalised, data_reflco_TM[2, 1, idx_wl, :], linestyle='dashed')
ax11.plot(lay_list_normalised, data_reflco_TM[3, 1, idx_wl, :])
ax11.set_title(r'$r_{TM}$, imag')
ax11.set_ylim((-1, 1))
plt.axes(ax11)
plt.legend(label_list)
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax02.plot(lay_list_normalised, data_SM[0, 0, idx_wl, :])
ax02.plot(lay_list_normalised, data_SM[1, 0, idx_wl, :])
ax02.plot(lay_list_normalised, data_SM[2, 0, idx_wl, :], linestyle='dashed')
ax02.plot(lay_list_normalised, data_SM[3, 0, idx_wl, :])
ax02.set_title(r'SM, real')
ax02.set_ylim((-1, 1))
plt.axes(ax02)
plt.legend(('S[0, 0]', 'S[0, 1]', 'S[1, 0]', 'S[1, 1]'))
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax03.plot(lay_list_normalised, data_SM[0, 1, idx_wl, :])
ax03.plot(lay_list_normalised, data_SM[1, 1, idx_wl, :])
ax03.plot(lay_list_normalised, data_SM[2, 1, idx_wl, :], linestyle='dashed')
ax03.plot(lay_list_normalised, data_SM[3, 1, idx_wl, :])
ax03.set_title(r'SM, imag')
ax03.set_ylim((-1, 1))
plt.axes(ax03)
plt.legend(('S[0, 0]', 'S[0, 1]', 'S[1, 0]', 'S[1, 1]'))
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax12.plot(lay_list_normalised, data_TM[0, 0, idx_wl, :])
ax12.plot(lay_list_normalised, data_TM[1, 0, idx_wl, :])
ax12.plot(lay_list_normalised, data_TM[2, 0, idx_wl, :], linestyle='dashed')
ax12.plot(lay_list_normalised, data_TM[3, 0, idx_wl, :])
ax12.set_title(r'TM, real')
ax12.set_ylim((-1000, 1000))
plt.axes(ax12)
plt.legend(('T[0, 0]', 'T[0, 1]', 'T[1, 0]', 'T[1, 1]'))
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

ax13.plot(lay_list_normalised, data_TM[0, 1, idx_wl, :])
ax13.plot(lay_list_normalised, data_TM[1, 1, idx_wl, :])
ax13.plot(lay_list_normalised, data_TM[2, 1, idx_wl, :], linestyle='dashed')
ax13.plot(lay_list_normalised, data_TM[3, 1, idx_wl, :])
ax13.set_title(r'TM, imag')
ax13.set_ylim((-1000, 1000))
plt.axes(ax13)
plt.legend(('T[0, 0]', 'T[0, 1]', 'T[1, 0]', 'T[1, 1]'))
plt.xlabel('Number of periods')
plt.xlim(xlimlist)

plt.tight_layout()

#fig.savefig("figure_7_a_analysis_chole_reflco.png", dpi=300)


#########################################
# Plot selected SM and TM matrix elements
step = 1
stop = N_hel360 * reso
print(max(lay_list[:stop:step]))
leglist = ('RCP to RCP', 'LCP to RCP', 'RCP to LCP', 'LCP to LCP')
idx_wl = 0
#xlimlist = (0, 100)
#leglist = ('p to p', 'p to s', 's to p', 's to s')
nb_layers = len(data_SM[0, 1, idx_wl, :])
colorlist = np.array(lay_list_normalised[:stop:step])
print(max(colorlist))

axislim = [-1.3, 1.3, -1.3, 1.3]
#axislim_TM = [-1.3e5, 1.3e5, -1.3e5, 1.3e5]
axislim_TM = [-1.3e7, 1.3e7, -1.3e7, 1.3e7]

common_cmap = 'viridis_r'
common_norm = plt.Normalize(0, max(colorlist))
common_size = 8

fig = plt.figure(constrained_layout=False, figsize=(4.5, 5.5))
#plt.tight_layout()
widths = [1, 1]
heights = [1, 1, 0.1]
gs = fig.add_gridspec(3, 2, width_ratios=widths, height_ratios=heights)
ax_00 = fig.add_subplot(gs[0, 0])
ax_10 = fig.add_subplot(gs[1, 0])
ax_01 = fig.add_subplot(gs[0, 1])
ax_11 = fig.add_subplot(gs[1, 1])
ax_4 = fig.add_subplot(gs[2, 0:2])

plt.axes(ax_00)
sca = plt.scatter(data_SM[0, 0, idx_wl, :stop:step], data_SM[0, 1, idx_wl, :stop:step], c=colorlist, cmap=common_cmap, norm=common_norm, s=common_size) #, edgecolors='k', linewidth=0.5)
plt.axis('equal')
plt.axis(axislim)
plt.xlabel('real')
plt.ylabel('imag')
plt.title(r'SM[0, 0]')

plt.axes(ax_01)
plt.scatter(data_SM[8, 0, idx_wl, :stop:step], data_SM[8, 1, idx_wl, :stop:step], c=colorlist, cmap=common_cmap, norm=common_norm, s=common_size)
plt.axis('equal')
plt.axis(axislim)
plt.xlabel('real')
plt.ylabel('imag')
plt.title(r'SM[2, 0]')

plt.axes(ax_10)
plt.scatter(data_TM[0, 0, idx_wl, :stop:step], data_TM[0, 1, idx_wl, :stop:step], c=colorlist, cmap=common_cmap, norm=common_norm, s=common_size)
plt.axis('equal')
plt.axis(axislim)
plt.xlabel('real')
plt.ylabel('imag')
plt.title(r'TM[0, 0]')
ax_10.ticklabel_format(style='sci', scilimits=(0, 2))
ax_10_inset = inset_axes(ax_10, width="30%", height="30%", loc=5)
plt.axes(ax_10_inset)
plt.scatter(data_TM[0, 0, idx_wl, :stop:step], data_TM[0, 1, idx_wl, :stop:step], c=colorlist, cmap=common_cmap, norm=common_norm, s=1)
#plt.axis('equal')
plt.axis(axislim_TM)
ax_10_inset.ticklabel_format(style='sci', scilimits=(0, 2))

plt.axes(ax_11)
plt.scatter(data_TM[8, 0, idx_wl, :stop:step], data_TM[8, 1, idx_wl, :stop:step], c=colorlist, cmap=common_cmap, norm=common_norm, s=common_size)
plt.axis('equal')
plt.axis(axislim)
plt.xlabel('real')
plt.ylabel('imag')
plt.title(r'TM[2, 0]')
ax_11.ticklabel_format(style='sci', scilimits=(0, 2))
ax_11_inset = inset_axes(ax_11, width="30%", height="30%", loc=5)
plt.axes(ax_11_inset)
plt.scatter(data_TM[8, 0, idx_wl, :stop:step], data_TM[8, 1, idx_wl, :stop:step], c=colorlist, cmap=common_cmap, norm=common_norm, s=1)
#plt.axis('equal')
plt.axis(axislim_TM)
ax_11_inset.ticklabel_format(style='sci', scilimits=(0, 2))

plt.axes(ax_4)
cb = plt.colorbar(sca, orientation='horizontal', cax=ax_4)
cb.set_ticks([1, 50, max(colorlist)])
cb.set_ticklabels([str(reso) + ' layers \n= 1 ' + r'$p$', str(50 * reso) + ' layers \n= 50 ' + r'$p$',  str(int(max(colorlist))) + r' $p$'])
cb.set_label('Number of layers')
#ax_4.imshow([colorlist], extent=[min(colorlist), max(colorlist), 0, 1], cmap='viridis_r')
#ax_4.set_yticks([])
#ax_4.set_xlim(0, max(colorlist))
#ax_4.set_title('Thickness (number of pitches)')

plt.tight_layout()

#fig.savefig("figure_7_a_analysis_chole_reflco_SM_TM_matrix.png", dpi=300)


#################################
# Plot all the SM matrix elements

idx_wl = 0
nb_layers = len(data_SM[0, 1, idx_wl, :])
step = 1
stop = 20 * reso
axislim = [-1.3, 1.3, -1.3, 1.3]
axislim_TM = [-1.3e5, 1.3e5, -1.3e5, 1.3e5]
common_cmap = 'viridis_r'
colourlist = np.array(lay_list[:stop:step]) ** 1

fig = plt.figure(constrained_layout=False, figsize=(7, 6))
widths = [1, 1, 1, 1]
heights = [1, 1, 1, 1]
gs = fig.add_gridspec(4, 4, width_ratios=widths, height_ratios=heights)
ax_00 = fig.add_subplot(gs[0, 0])
ax_01 = fig.add_subplot(gs[0, 1])
ax_02 = fig.add_subplot(gs[0, 2])
ax_03 = fig.add_subplot(gs[0, 3])
ax_10 = fig.add_subplot(gs[1, 0])
ax_11 = fig.add_subplot(gs[1, 1])
ax_12 = fig.add_subplot(gs[1, 2])
ax_13 = fig.add_subplot(gs[1, 3])
ax_20 = fig.add_subplot(gs[2, 0])
ax_21 = fig.add_subplot(gs[2, 1])
ax_22 = fig.add_subplot(gs[2, 2])
ax_23 = fig.add_subplot(gs[2, 3])
ax_30 = fig.add_subplot(gs[3, 0])
ax_31 = fig.add_subplot(gs[3, 1])
ax_32 = fig.add_subplot(gs[3, 2])
ax_33 = fig.add_subplot(gs[3, 3])
plot_all_blocks(data_SM)
plt.suptitle('SM')
plt.tight_layout()
#fig.savefig("figure_7_a_analysis_chole_reflco_matrix_elements_SM.png", dpi=300)


#################################
# Plot all the TM matrix elements

idx_wl = 0
nb_layers = len(data_SM[0, 1, idx_wl, :])
step = 1
stop = 20 * reso
axislim = [-1.3, 1.3, -1.3, 1.3]
axislim_TM = [-1.3e5, 1.3e5, -1.3e5, 1.3e5]
common_cmap = 'viridis_r'
colourlist = np.array(lay_list[:stop:step]) ** 1

fig = plt.figure(constrained_layout=False, figsize=(7, 6))
widths = [1, 1, 1, 1]
heights = [1, 1, 1, 1]
gs = fig.add_gridspec(4, 4, width_ratios=widths, height_ratios=heights)
ax_00 = fig.add_subplot(gs[0, 0])
ax_01 = fig.add_subplot(gs[0, 1])
ax_02 = fig.add_subplot(gs[0, 2])
ax_03 = fig.add_subplot(gs[0, 3])
ax_10 = fig.add_subplot(gs[1, 0])
ax_11 = fig.add_subplot(gs[1, 1])
ax_12 = fig.add_subplot(gs[1, 2])
ax_13 = fig.add_subplot(gs[1, 3])
ax_20 = fig.add_subplot(gs[2, 0])
ax_21 = fig.add_subplot(gs[2, 1])
ax_22 = fig.add_subplot(gs[2, 2])
ax_23 = fig.add_subplot(gs[2, 3])
ax_30 = fig.add_subplot(gs[3, 0])
ax_31 = fig.add_subplot(gs[3, 1])
ax_32 = fig.add_subplot(gs[3, 2])
ax_33 = fig.add_subplot(gs[3, 3])
plot_all_blocks(data_TM)
plt.suptitle('TM')
plt.tight_layout()
#fig.savefig("figure_7_a_analysis_chole_reflco_matrix_elements_TM.png", dpi=300)


plt.show()




