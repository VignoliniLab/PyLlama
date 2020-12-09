import pyllama as sc
import numpy as np
import cholesteric as ch
import matplotlib.pyplot as plt


"""
This script calculates the reflection spectra (RCP, LCP, unpolarised) for:
    - a right-handed cholesteric
    - a left-handed cholesteric
    - a tilted cholesteric under normal illumination
    - a cholesteric under oblique illumination
    - a compressed cholesteric
    - a distorted cholesteric
The user is invited to mix and match these examples.
The output of this script is used for Figure 5.
"""


# Definition of the functions that construct the plots and calculate the spectra
def custom_plot_sche(ch, fig_f, ax_sche):
    """
    Axis with the 3D schematic
    """
    _, _ = ch.plot_simple(fig_f, ax_sche, view='classic', type='3D')
    ch.plot_add_optics(fig_f, ax_sche, theta_in_rad)
    ax_sche.xaxis.set_major_locator(plt.NullLocator())
    ax_sche.yaxis.set_major_locator(plt.NullLocator())
    ax_sche.set_zlim3d([0, 500])
    ax_sche.set_zticks([0, 250, 500])
    ax_sche.set_xlabel("")
    ax_sche.set_ylabel("")
    ax_sche.set_zlabel("")


def custom_plot_plo(sp, ax_plo):
    """
    Axis with the reflectance plots + calculation of the reflectance
    """
    sp.calculate_refl_trans(circ=True, method="TM", talk=True)  # TM = faster
    ax_plo.plot(wl_nm_list, 0.5 * (sp.data['R_R_to_R'] + sp.data['R_L_to_R']), label="RCP")
    ax_plo.plot(wl_nm_list, 0.5 * (sp.data['R_L_to_R'] + sp.data['R_L_to_L']), label="LCP")
    ax_plo.plot(wl_nm_list, 0.5 * (sp.data['R_R_to_R'] + sp.data['R_R_to_L'] + sp.data['R_L_to_R'] + sp.data['R_L_to_L']),
                color='k', linestyle='dotted', label="unpolarised")
    ax_plo.set_ylim((0, 1))
    plt.yticks((0, 1))
    ax_plo.set_xlim((400, 800))
    plt.xticks((400, 600, 800))
    ax_plo.set_xlabel(r'$\lambda$ (nm)')
    ax_plo.set_ylabel("Reflectance")


def make_and_save_fig(title_string, file_string, chole_small_f, spectrum_f):
    """
    Figure with the 3D schematic and the reflectance plot
    """
    fig = plt.figure(constrained_layout=False, figsize=(7, 3))  # paper: (3.8, 1.7)
    plt.tight_layout()
    widths = [2, 0.1, 2]
    heights = [0.1, 2, 0.1]
    gs = fig.add_gridspec(3, 3, width_ratios=widths, height_ratios=heights)
    ax_sche = fig.add_subplot(gs[:, 0], projection='3d', proj_type='ortho')
    ax_plot = fig.add_subplot(gs[:-1, 2])
    custom_plot_sche(chole_small_f, fig, ax_sche)
    custom_plot_plo(spectrum_f, ax_plot)
    ax_plot.set_title(title_string)
    ax_plot.legend(ncol=1)  # paper: 3
    ax_sche.view_init(180 + 10, 180 + 70)
    plt.tight_layout(h_pad=0, w_pad=0)
    fig.savefig(file_string, dpi=300)


# Parameters to choose manually
n_av = 1.433
biref = 0.2  # a very high birefringence emphasises the differences
n_e = n_av + 0.5 * biref
n_o = n_av - 0.5 * biref
n_entry = n_av
n_exit = n_av
pitch_nm = 500
N_hel360 = 10
wl_min = 400
wl_max = 800

# Parameters that are calculated automatically
wl_nm_list = np.linspace(start=wl_min, stop=wl_max, num=1 * (wl_max - wl_min), endpoint=False)

# RIGHT-HANDED CHOLESTERIC
theta_in_deg = 0
tilt_deg = 0
periods = 1
handedness = 1
theta_in_rad = theta_in_deg * np.pi / 180
tilt_rad = tilt_deg * np.pi / 180
chole_small = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=12)
chole_big = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=40)
spectrum = sc.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole_big, n_e=n_e, n_o=n_o, n_entry=n_entry,
                                                       n_exit=n_exit, N_per=N_hel360, theta_in_rad=theta_in_rad))
make_and_save_fig("Right-handed", "chole_showcase_right.svg", chole_small, spectrum)
print("Usual cholesteric done.")


# TILTED CHOLESTERIC
theta_in_deg = 0
tilt_deg = 30
periods = 1
handedness = 1
theta_in_rad = theta_in_deg * np.pi / 180
tilt_rad = tilt_deg * np.pi / 180
chole_small = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=12)
chole_big = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=40)
print("hereeeeeeeeeeee")
print(chole_big.helical_axis)
spectrum = sc.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole_big, n_e=n_e, n_o=n_o, n_entry=n_entry,
                                                       n_exit=n_exit, N_per=N_hel360, theta_in_rad=theta_in_rad))
make_and_save_fig("Tilted", "chole_showcase_tilted.svg", chole_small, spectrum)
print("Tilted cholesteric done.")


# COMPRESSED CHOLESTERIC
pitch_nm_compressed = pitch_nm
theta_in_deg = 0
tilt_deg = 0
periods = 1
handedness = 1
theta_in_rad = theta_in_deg * np.pi / 180
tilt_rad = tilt_deg * np.pi / 180
chole_small = ch.Cholesteric(pitch360=pitch_nm_compressed, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=12)
chole_small.compress(0.7)
chole_big = ch.Cholesteric(pitch360=pitch_nm_compressed, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=20)
print([d * 180 / np.pi for d in chole_big.slices_rotangles])
chole_big.compress(0.7)
print([d * 180 / np.pi for d in chole_big.slices_rotangles])
spectrum = sc.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole_big, n_e=n_e, n_o=n_o, n_entry=n_entry,
                                                       n_exit=n_exit, N_per=N_hel360, theta_in_rad=theta_in_rad))
make_and_save_fig("Compressed", "chole_showcase_compressed.svg", chole_small, spectrum)
print("Compressed cholesteric done.")


# LEFT-HANDED CHOLESTERIC
theta_in_deg = 0
tilt_deg = 0
periods = 1
handedness = -1
theta_in_rad = theta_in_deg * np.pi / 180
tilt_rad = tilt_deg * np.pi / 180
chole_small = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=12)
chole_big = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=40)
spectrum = sc.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole_big, n_e=n_e, n_o=n_o, n_entry=n_entry,
                                                       n_exit=n_exit, N_per=N_hel360, theta_in_rad=theta_in_rad))
make_and_save_fig("Left-handed", "chole_showcase_left.svg", chole_small, spectrum)
print("Left-handed cholesteric done.")


# OBLIQUE INCIDENCE
theta_in_deg = 70
tilt_deg = 30
periods = 1
handedness = 1
theta_in_rad = theta_in_deg * np.pi / 180
tilt_rad = tilt_deg * np.pi / 180
chole_small = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=12)
chole_big = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=40)
spectrum = sc.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole_big, n_e=n_e, n_o=n_o, n_entry=n_entry,
                                                       n_exit=n_exit, N_per=N_hel360, theta_in_rad=theta_in_rad))
make_and_save_fig("Oblique", "chole_showcase_oblique.svg", chole_small, spectrum)
print("Oblique incidence cholesteric done.")


# DISTORTED
theta_in_deg = 0
tilt_deg = 0  #80
periods = 1
handedness = 1
theta_in_rad = theta_in_deg * np.pi / 180
tilt_rad = tilt_deg * np.pi / 180
chole_small = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=12)
chole_small.distort(3)
chole_big = ch.Cholesteric(pitch360=pitch_nm, tilt_rad=tilt_rad, N_hel360=periods, handedness=handedness, resolution=20)
print([d * 180 / np.pi for d in chole_big.slices_rotangles])
chole_big.distort(3)
print([d * 180 / np.pi for d in chole_big.slices_rotangles])
spectrum = sc.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole_big, n_e=n_e, n_o=n_o, n_entry=n_entry,
                                                       n_exit=n_exit, N_per=N_hel360, theta_in_rad=theta_in_rad))
make_and_save_fig("Distorted", "chole_showcase_distorted.svg", chole_small, spectrum)
print("Distorted cholesteric done.")


# Plot the results
plt.show()


