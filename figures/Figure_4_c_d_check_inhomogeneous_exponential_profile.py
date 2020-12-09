import numpy as np
import scipy as sp
import scipy.special
import matplotlib.pyplot as plt
import pyllama as ll
from itertools import accumulate
import matplotlib.pylab as pl


"""
The systems modelled in this script are:
    1. an inhomogeneous slab where the refractive index has an exponential profile (index-matched at the entry and exit media)
    2. an inhomogeneous slab where the refractive index has an hyperbolic profile (index-matched at the entry and exit media)
P. Yeh (Optical Waves in Layered Media, ISBN: 978-0-471-73192-4, chapter 8) provides analytical formulas for the 
modelling of such inhomogeneous slabs. He also explains how to decompose these slabs into a multilayered stack:
    3. for the exponential profile: small layers of equal thickness, the refractive indices of the layers vary exponentially
    4. for the hyperbolic profile:
        4.1. small layers of equal thickness, the refractive indices of the layers vary in a hyperbolic way
        4.2. small layers of equal optical thickness, the refractive indices of the layers vary in an exponential way
In this script we represent:
    - with P. Yeh’s analytical formulas: situations 1 and 2
    - with P. Yeh’s discrete multilayer model: situation 4.2
    - with PyLlama and a discrete multilayer model: situations 3 and 4.2
The outputs of this script, for the cases 1 and 3, are used in Figures 4c-d.
"""


# Bessel functions
def Y0(x):
    return sp.special.yv(0, x)


def Y1(x):
    return sp.special.yv(1, x)


def J0(x):
    return sp.special.jv(0, x)


def J1(x):
    return sp.special.jv(1, x)


# Yeh, situation 1
def refl_yeh_continuous_exponential(n0_f, ns_f, L_f, wl_f):
    y0 = 2 * np.pi * n0_f * L_f / (wl_f * np.log(ns_f / n0_f))
    ys = 2 * np.pi * ns_f * L_f / (wl_f * np.log(ns_f / n0_f))
    A = J0(y0) - 1j * J1(y0)
    B = Y0(ys) - 1j * Y1(ys)
    C = Y0(y0) - 1j * Y1(y0)
    D = J0(ys) - 1j * J1(ys)
    E = J0(y0) + 1j * J1(y0)
    F = Y0(ys) - 1j * Y1(ys)
    G = Y0(y0) + 1j * Y1(y0)
    H = J0(ys) - 1J * J1(ys)
    nume = A * B - C * D
    deno = E * F - G * H
    r = nume / deno
    refl = np.abs(r) ** 2
    return refl


# Yeh, situation 2
def refl_yeh_continuous_hyperbolic(n0_f, ns_f, L_f, wl_f):
    phi_cap = 2 * np.pi * n0_f * ns_f * L_f * np.log(ns_f / n0_f) / (wl_f * (ns_f - n0_f))
    delta_cap_sq = 0.25 * (np.log(ns_f / n0_f)) ** 2
    temp = phi_cap ** 2 - delta_cap_sq
    refl = (1 + (temp / (delta_cap_sq * (np.sin(np.sqrt(temp))) ** 2))) ** -1
    return refl


# For PyLlama: refractive indices for layers that have the same thickness
def index_discrete_same_thickness(n0_f, ns_f, L_f, resolution_f):
    thickness_list_same = [L_f / (resolution_f + 1) for lay in range(0, resolution_f + 1, 1)]
    temp = [0] + thickness_list_same
    thickness_list_cumulative = [sum(temp[0:k + 1]) for k in range(len(temp) - 1)]
    n_list_same = [n0_f * (ns_f / n0_f) ** (x / L_f) for x in thickness_list_cumulative]
    n_list_same.append(ns_f)
    eps_list_same = [np.array([[ni ** 2, 0, 0], [0, ni ** 2, 0], [0, 0, ni ** 2]]) for ni in n_list_same]
    thickness_list_same.append(L_f)
    return n_list_same, eps_list_same, thickness_list_same


# For PyLlama: refractive indices for layers that have the same optical thickness
def index_discrete_optical_thickness(n0_f, ns_f, L_f, wl_f, resolution_f):
    # Calculate the refractive index of the layers
    layers = range(0, resolution_f + 1, 1)
    n_list_optical = [n0_f * (ns_f / n0_f) ** (lay / (resolution_f + 1)) for lay in layers]
    # Calculate the thickness of the layers
    phi = 1
    thickness_list_1 = [wl_f * phi / (2 * np.pi * ni) for ni in n_list_optical]
    thickness_list_optical = [t * L_f / sum(thickness_list_1) for t in thickness_list_1]
    phi_correc = 2 * np.pi * n_list_optical[0] * thickness_list_optical[0] / wl_f
    # Adding last layer + permittivity
    n_list_optical.append(ns_f)
    thickness_list_optical.append(L / sum(thickness_list_optical))
    eps_list_optical = [np.array([[ni ** 2, 0, 0], [0, ni ** 2, 0], [0, 0, ni ** 2]]) for ni in n_list_optical]
    return n_list_optical, eps_list_optical, thickness_list_optical, phi_correc


# Yeh, situation 4.2
def refl_yeh_discrete_optical_thickness(n0_f, ns_f, L_f, wl_f, resolution_f):
    n_list_optical, eps_list_optical, thickness_list_optical, phi_correc = index_discrete_optical_thickness(n0_f, ns_f, L_f, wl_f, resolution_f)

    # Calculate the reflection spectrum with Yeh for DISCRETE exponential layers
    resolution_correc = len(n_list_optical) - 2  # don't count the first and last layers
    beta = (ns_f / n0_f) ** (1 / (resolution_correc + 1))
    y = np.arccos(np.cos(phi_correc) * (1 + beta) / (2 * np.sqrt(beta)))
    C = ((1 - beta) / (2 * np.sqrt(beta))) * np.exp(1j * phi_correc)
    refl = (1 + ((np.sin(y)) ** 2) / ((np.abs(C) ** 2) * (np.sin(y * (resolution_correc + 1))) ** 2)) ** -1
    return refl


# PyLlama, situation 4.2
def refl_pyllama_optical_thickness(n0_f, ns_f, L_f, wl_f, resolution_f):
    n_entry = n0_f
    n_exit = ns_f
    theta_in_rad = 0

    n_list_optical, eps_list_optical, thickness_list_optical, _ = index_discrete_optical_thickness(n0_f, ns_f, L_f, wl_f, resolution_f)
    # Calculate the reflection spectrum with PyLlama for layers of optical thickness
    #model = ll.StackModel(eps_list_optical, thickness_list_optical, n_entry, n_exit, wl_f, theta_in_rad)
    model = ll.StackOpticalThicknessModel(n_list_optical[:-1:], L_f, n_entry, n_exit, wl_f, theta_in_rad)
    refl_mat, trans_mat = model.get_refl_trans(method='SM', circ=False)
    refl = float(refl_mat[0, 0])
    return refl


# PyLlama, situation 3
def refl_pyllama_same_thickness(n0_f, ns_f, L_f, wl_f, resolution_f):
    n_list_same, eps_list_same, thickness_list_same = index_discrete_same_thickness(n0_f, ns_f, L_f, resolution_f)

    n_entry = n0_f
    n_exit = ns_f
    theta_in_rad = 0

    model = ll.StackModel(eps_list_same, thickness_list_same, n_entry, n_exit, wl_f, theta_in_rad)
    refl_mat, trans_mat = model.get_refl_trans(method='SM', circ=False)
    refl = float(refl_mat[0, 0])
    return refl


# Calculation of the reflectance for all methods
def calculation_all_spectra(xvar_list, resolution):
    refl_yeh_discrete = []
    refl_yeh_continuous = []
    refl_yeh_hyper = []
    refl_pyllama_optical = []
    refl_pyllama_same = []
    wl_list = []
    iter_xvar = 0
    for xvar in xvar_list:
        # Calculate the wavelength
        wl = L / xvar
        wl_list.append(wl)

        # Calculate the reflection spectrum with Yeh
        refl = refl_yeh_continuous_exponential(n0, ns, L, wl)
        refl_yeh_continuous.append(refl)

        # Calculate the reflection spectrum with Yeh for DISCRETE exponential layers
        refl = refl_yeh_discrete_optical_thickness(n0, ns, L, wl, resolution)
        refl_yeh_discrete.append(refl)

        # Calculate the reflection spectrum with PyLlama for layers of optical thickness
        refl = refl_pyllama_optical_thickness(n0, ns, L, wl, resolution)
        refl_pyllama_optical.append(refl)

        # Calculate the reflection spectrum with PyLlama for layers of same thickness
        refl = refl_pyllama_same_thickness(n0, ns, L, wl, resolution)
        refl_pyllama_same.append(refl)

        # Calculate the hyperbolic reflection spectrum with Yeh
        refl = refl_yeh_continuous_hyperbolic(n0, ns, L, wl)
        refl_yeh_hyper.append(refl)

        iter_xvar = iter_xvar + 1
    return [wl_list, refl_yeh_continuous, refl_yeh_hyper, refl_pyllama_optical, refl_pyllama_same, refl_yeh_discrete]


# Parameters for all
n0 = 1
ns = 4
L = 100  # total thickness of the layer in nm
xvar_list = np.linspace(0.01, 1, 100)

# Initialisation of figures
fig_index_expo = plt.figure(0, constrained_layout=False)  # papprE figsize=(3, 2)
widths = [1]
heights = [1]
gs = fig_index_expo.add_gridspec(1, 1, width_ratios=widths, height_ratios=heights)
ax_index_expo = fig_index_expo.add_subplot(gs[0, 0])
plt.title("Exponential")

fig_spec_expo = plt.figure(1, constrained_layout=False)  # papprE figsize=(3, 2)
widths = [1]
heights = [1]
gs = fig_spec_expo.add_gridspec(1, 1, width_ratios=widths, height_ratios=heights)
ax_spec_expo = fig_spec_expo.add_subplot(gs[0, 0])
plt.title("Exponential")

fig_index_hyper = plt.figure(2, constrained_layout=False)  # papprE figsize=(3, 2)
widths = [1]
heights = [1]
gs = fig_index_hyper.add_gridspec(1, 1, width_ratios=widths, height_ratios=heights)
ax_index_hyper = fig_index_hyper.add_subplot(gs[0, 0])
plt.title("Hyperbolic")

fig_spec_hyper = plt.figure(3, constrained_layout=False)  # papprE figsize=(3, 2)
widths = [1]
heights = [1]
gs = fig_spec_hyper.add_gridspec(1, 1, width_ratios=widths, height_ratios=heights)
ax_spec_hyper = fig_spec_hyper.add_subplot(gs[0, 0])
plt.title("Hyperbolic")

# Calculation in two resolutions
lowres = 4
result = calculation_all_spectra(xvar_list, lowres)
wl_list = result[0]
refl_yeh_continuous_lowres = result[1]
refl_yeh_hyper_lowres = result[2]
refl_pyllama_optical_lowres = result[3]
refl_pyllama_same_lowres = result[4]
refl_yeh_discrete_lowres = result[5]

highres = 20
result = calculation_all_spectra(xvar_list, highres)
wl_list_highres = result[0]
refl_yeh_continuous_highres = result[1]
refl_yeh_hyper_highres = result[2]
refl_pyllama_optical_highres = result[3]
refl_pyllama_same_highres = result[4]
refl_yeh_discrete_highres = result[5]

# Set the colours
colors = pl.cm.tab10(np.linspace(0, 1, 10))

# Calculate all the indices to plot for the low resolution
resolution = lowres
zaxis_layer = np.linspace(0, L, 100)
n_expo = [n0 * np.exp((z / L) * np.log(ns / n0)) for z in zaxis_layer]
n_hyper = [n0 / (1 - ((ns - n0) / ns) * (z / L)) for z in zaxis_layer]
list_indices_optical, _, list_thickness_optical, _ = index_discrete_optical_thickness(n0, ns, L, wl_list[0], resolution)
list_indices_same, _, list_thickness_same = index_discrete_same_thickness(n0, ns, L, resolution)
list_thickness_optical_cumulative = [0] + list(accumulate(list_thickness_optical[:-1]))
list_thickness_same_cumulative = [0] + list(accumulate(list_thickness_same[:-1]))
for k in range(0, len(list_thickness_optical_cumulative) - 1):
    ax_index_expo.plot([list_thickness_same_cumulative[k], list_thickness_same_cumulative[k+1]],
              [list_indices_same[k], list_indices_same[k]], color=colors[2], label=r'Same $h$')
    ax_index_expo.plot([list_thickness_same_cumulative[k + 1], list_thickness_same_cumulative[k + 1]],
              [list_indices_same[k], list_indices_same[k + 1]], color=colors[2], label=r'Same $h$')
    ax_index_hyper.plot([list_thickness_optical_cumulative[k], list_thickness_optical_cumulative[k + 1]],
              [list_indices_optical[k], list_indices_optical[k]], color=colors[2], label=r'Same $h$')
    ax_index_hyper.plot([list_thickness_optical_cumulative[k + 1], list_thickness_optical_cumulative[k + 1]],
              [list_indices_optical[k], list_indices_optical[k + 1]], color=colors[2], label=r'Same $h$')

# Calculate all the indices to plot for the high resolution
resolution = highres
zaxis_layer = np.linspace(0, L, 100)
n_expo = [n0 * np.exp((z / L) * np.log(ns / n0)) for z in zaxis_layer]
n_hyper = [n0 / (1 - ((ns - n0) / ns) * (z / L)) for z in zaxis_layer]
list_indices_optical, _, list_thickness_optical, _ = index_discrete_optical_thickness(n0, ns, L, wl_list[0], resolution)
list_indices_same, _, list_thickness_same = index_discrete_same_thickness(n0, ns, L, resolution)
list_thickness_optical_cumulative = [0] + list(accumulate(list_thickness_optical[:-1]))
list_thickness_same_cumulative = [0] + list(accumulate(list_thickness_same[:-1]))
for k in range(0, len(list_thickness_optical_cumulative) - 1):
    ax_index_expo.plot([list_thickness_same_cumulative[k], list_thickness_same_cumulative[k+1]],
              [list_indices_same[k], list_indices_same[k]], color=colors[1], label=r'Same $h$')
    ax_index_expo.plot([list_thickness_same_cumulative[k + 1], list_thickness_same_cumulative[k + 1]],
              [list_indices_same[k], list_indices_same[k + 1]], color=colors[1], label=r'Same $h$')
    ax_index_hyper.plot([list_thickness_optical_cumulative[k], list_thickness_optical_cumulative[k + 1]],
              [list_indices_optical[k], list_indices_optical[k]], color=colors[1], label=r'Same $h$')
    ax_index_hyper.plot([list_thickness_optical_cumulative[k + 1], list_thickness_optical_cumulative[k + 1]],
              [list_indices_optical[k], list_indices_optical[k + 1]], color=colors[1], label=r'Same $h$')
ax_index_expo.plot(zaxis_layer, n_expo, color=colors[0], linestyle="solid", label="Exponential")
ax_index_hyper.plot(zaxis_layer, n_hyper, color=colors[0], linestyle="solid", label="Hyperbolic")
ax_index_expo.set_yticks([1, 4])
ax_index_expo.set_yticklabels([r'$n_0$', r'$n_s$'])
ax_index_hyper.set_yticks([1, 4])
ax_index_hyper.set_yticklabels([r'$n_0$', r'$n_s$'])

# Plot the indices for the exponential case
plt.figure(0)
plt.plot([-L/3, 0], [n0, n0], color='k', linestyle='dashed', label="Half-spaces")
plt.plot([L, L + L/3], [ns, ns], color='k', linestyle='dashed')
plt.plot([0, 0], [n0-0.2, ns+0.2], color=[0.5, 0.5, 0.5], linestyle='dotted')
plt.plot([L, L], [n0-0.2, ns+0.2], color=[0.5, 0.5, 0.5], linestyle='dotted')
plt.xlabel('z axis')
plt.ylabel('Refractive index')
plt.tight_layout()

# Plot the indices for the hyperbolic case
plt.figure(2)
plt.plot([-L/3, 0], [n0, n0], color='k', linestyle='dashed', label="Half-spaces")
plt.plot([L, L + L/3], [ns, ns], color='k', linestyle='dashed')
plt.plot([0, 0], [n0-0.2, ns+0.2], color=[0.5, 0.5, 0.5], linestyle='dotted')
plt.plot([L, L], [n0-0.2, ns+0.2], color=[0.5, 0.5, 0.5], linestyle='dotted')
plt.xlabel('z axis')
plt.ylabel('Refractive index')
plt.tight_layout()

# Plot the reflectances for the exponential case
ax_spec_expo.plot(xvar_list, refl_yeh_continuous_highres, color=colors[0], label="Continuous, Yeh")
ax_spec_expo.plot(xvar_list, refl_pyllama_same_lowres, color=colors[2], linestyle="dashed", label=r'PyLlama, N = 4')
ax_spec_expo.plot(xvar_list, refl_pyllama_same_highres, color=colors[1], linestyle="dashed", label=r'PyLlama, N = 20')

# Plot the reflectances for the hyperbolic case
ax_spec_hyper.plot(xvar_list, refl_yeh_hyper_highres, color=colors[0], label="Continuous, Yeh")
ax_spec_hyper.plot(xvar_list, refl_pyllama_optical_lowres, color=colors[2], linestyle="dashed", label=r'PyLlama, N = 4')
ax_spec_hyper.plot(xvar_list, refl_pyllama_optical_highres, color=colors[1], linestyle="dashed", label=r'PyLlama, N = 20')
ax_spec_hyper.plot(xvar_list, refl_yeh_discrete_lowres, color=colors[3], linestyle="dotted", label="Discrete, Yeh, N = 4")
ax_spec_hyper.plot(xvar_list, refl_yeh_discrete_highres, color=colors[4], linestyle="dotted", label="Discrete, Yeh, N = 20")

# General set-up
plt.figure(1)
plt.xlabel(r'$h / \lambda$')
plt.ylabel("Reflectance")
plt.xlim((0, 1))
plt.legend()
plt.tight_layout()
plt.figure(3)
plt.xlabel(r'$h / \lambda$')
plt.ylabel("Reflectance")
plt.xlim((0, 1))
plt.legend()
plt.tight_layout()

#fig_index_expo.savefig('fig_yeh_non_homogeneous_index_expo.png', dpi=300)
#fig_spec_expo.savefig('fig_yeh_non_homogeneous_spec_expo.png', dpi=300)
#fig_index_hyper.savefig('fig_yeh_non_homogeneous_index_hyper.png', dpi=300)
#fig_spec_hyper.savefig('fig_yeh_non_homogeneous_spec_hyper.png', dpi=300)

plt.show()


