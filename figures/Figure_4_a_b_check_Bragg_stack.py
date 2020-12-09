import pyllama as ll
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as np_la
import matplotlib.pylab as pl


"""
This script calculates the s and p reflection spectra of a Bragg stack with PyLlama and compare the results with 
formulas from Pochi Yeh. The outputs are used for Figure 4.
"""


def yeh_spectrum(N_stack, n_entry, n_exit, n_list_unit, thick_nm_list_unit, theta_deg_in, lbd_nm_list):
    n_list_stack = [n_entry] + n_list_unit * N_stack + [n_exit]
    thick_nm_list_stack = [0] + thick_nm_list_unit * N_stack + [0]
    theta_rad_in = theta_deg_in * np.pi / 180

    R_s = []
    R_p = []
    for lbd_nm in lbd_nm_list:
        M_s = np.eye(2)
        M_p = np.eye(2)
        for klay in range(0, len(n_list_stack)):
            n_lay = n_list_stack[klay]
            d_lay = thick_nm_list_stack[klay]
            theta_rad_lay = np.arcsin(n_list_stack[0] * np.sin(theta_rad_in) / n_lay)
            D_lay_s = get_dynamic_matrix(n_lay, theta_rad_lay, 's')
            D_lay_p = get_dynamic_matrix(n_lay, theta_rad_lay, 'p')
            P_lay = get_propa_matrix(n_lay, theta_rad_lay, d_lay, lbd_nm)
            if klay == 0:
                M_s = np.dot(np_la.inv(D_lay_s), M_s)
                M_p = np.dot(np_la.inv(D_lay_p), M_p)
            elif klay == len(n_list_stack) - 1:
                M_s = np.dot(M_s, D_lay_s)
                M_p = np.dot(M_p, D_lay_p)
            else:
                M_s = np_la.multi_dot((M_s, D_lay_s, P_lay, np_la.inv(D_lay_s)))
                M_p = np_la.multi_dot((M_p, D_lay_p, P_lay, np_la.inv(D_lay_p)))
        r_s = abs(M_s[1, 0] / M_s[0, 0]) ** 2
        r_p = abs(M_p[1, 0] / M_p[0, 0]) ** 2
        R_s.append(r_s)
        R_p.append(r_p)
    return R_s, R_p


def get_dynamic_matrix(n, theta_rad, pol):
    if pol == 's':
        D = np.array([[1, 1],
                      [n * np.cos(theta_rad), - n * np.cos(theta_rad)]
                      ])
    elif pol == 'p':
        D = np.array([[np.cos(theta_rad), np.cos(theta_rad)],
                      [n, -n]
                      ])
    return D


def get_propa_matrix(n, theta_rad, d, lbd_nm):
    phi = 2 * np.pi * n * d * np.cos(theta_rad) / lbd_nm
    P = np.array([[np.exp(1j * phi), 0],
                  [0, np.exp(-1j * phi)]])
    return P


def map(X, YMIN, YMAX, XMIN, XMAX):
    n1 = len(X)
    Y = []
    for i1 in range(0, n1):
        Y.append((YMAX - YMIN) * (X[i1] - XMIN) / (XMAX - XMIN) + YMIN)
    return Y


# Parameters to choose manually
n0 = 2.2
n1 = 1
n_entry = 1
n_exit = 2.2
thick0_nm = 200
thick1_nm = 500
thick_period_nm = thick0_nm + thick1_nm
N = 10
#theta_in_deg = 0
wl_min = 400
wl_max = 801

eps0 = np.array([
    [n0**2, 0, 0],
    [0, n0**2, 0],
    [0, 0, n0**2]
])
eps1 = np.array([
    [n1**2, 0, 0],
    [0, n1**2, 0],
    [0, 0, n1**2]
])

wl_nm_list = np.arange(wl_min, wl_max, 1)
theta_in_deg = 60
theta_in_rad = theta_in_deg * np.pi / 180

# Spectra with PyLlama
n_list = [n1, n0]
eps_list = [eps1, eps0]
thickness_list = [thick1_nm, thick0_nm]
data_sc_pp = []
data_sc_ss = []

for wl in wl_nm_list:
    model = ll.StackModel(eps_list, thickness_list, n_entry, n_exit, wl, theta_in_rad, N)
    refl, trans = model.get_refl_trans(method="SM", circ=False)
    refl_p_to_p = float(refl[0, 0])
    refl_s_to_s = float(refl[1, 1])
    data_sc_pp.append(refl_p_to_p)
    data_sc_ss.append(refl_s_to_s)

# Spectra with Yeh
data_yeh_fun_s, data_yeh_fun_p = yeh_spectrum(N, n_entry, n_exit, [n1, n0], [thick1_nm, thick0_nm], theta_in_deg, wl_nm_list)

# Plotting
fig_spec = plt.figure(3, constrained_layout=False)  # in the paper: figsize=(2.7, 1.8)
widths = [1]
heights = [1]
gs = fig_spec.add_gridspec(1, 1, width_ratios=widths, height_ratios=heights)
ax_spec = fig_spec.add_subplot(gs[0, 0])

colors = pl.cm.tab10(np.linspace(0, 1, 10))
ax_spec.plot(wl_nm_list, data_yeh_fun_p, color=colors[0], label="Analytical p")
ax_spec.plot(wl_nm_list, data_sc_pp, linestyle='dashed', color=colors[1], label="PyLlama p")
ax_spec.plot(wl_nm_list, data_yeh_fun_s, color=colors[2], label="Analytical s")
ax_spec.plot(wl_nm_list, data_sc_ss, linestyle='dashed', color=colors[3], label="PyLlama s")
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'Reflectance')
plt.legend(loc=1, prop={'size': 9})

plt.tight_layout()

#fig_spec.savefig('figure_4_a_b_Bragg_stack.png', dpi=300)

plt.show()


