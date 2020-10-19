import pyllama as ll
import numpy as np
import cholesteric as ch
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


"""
This script calculate the reflection and transmission spectra of a small and a big cholesteric domains using the
exponential, transfer and scattering matrix methods. The EM and TM methods are expected to fail for the "large" domain 
(large in terms of thickness and/or birefringence) while the SM method is expected to always work. This code will 
therefore display warnings for the EM and TM methods in the case of the large cholesteric domain, it is an expected 
behaviour to demonstrate the stability of the SM method.
"""


def calculate_all_spectra(spectrum):
    spectrum.calculate_refl_trans(circ=False, method="SM")
    spectrum.rename_result("R_pp", "R_pp_SM")
    spectrum.rename_result("R_ps", "R_ps_SM")
    spectrum.rename_result("R_sp", "R_sp_SM")
    spectrum.rename_result("R_ss", "R_ss_SM")
    spectrum.rename_result("T_pp", "T_pp_SM")
    spectrum.rename_result("T_ps", "T_ps_SM")
    spectrum.rename_result("T_sp", "T_sp_SM")
    spectrum.rename_result("T_ss", "T_ss_SM")
    spectrum.rename_result("time_elapsed", "time_elapsed_SM")
    print("SM linear basis done.")
    spectrum.calculate_refl_trans(circ=True, method="SM")
    spectrum.rename_result("R_RR", "R_RR_SM")
    spectrum.rename_result("R_RL", "R_RL_SM")
    spectrum.rename_result("R_LR", "R_LR_SM")
    spectrum.rename_result("R_LL", "R_LL_SM")
    spectrum.rename_result("T_RR", "T_RR_SM")
    spectrum.rename_result("T_RL", "T_RL_SM")
    spectrum.rename_result("T_LR", "T_LR_SM")
    spectrum.rename_result("T_LL", "T_LL_SM")
    print("SM circular basis done.")
    spectrum.calculate_refl_trans(circ=False, method="TM")
    spectrum.rename_result("R_pp", "R_pp_TM")
    spectrum.rename_result("R_ps", "R_ps_TM")
    spectrum.rename_result("R_sp", "R_sp_TM")
    spectrum.rename_result("R_ss", "R_ss_TM")
    spectrum.rename_result("T_pp", "T_pp_TM")
    spectrum.rename_result("T_ps", "T_ps_TM")
    spectrum.rename_result("T_sp", "T_sp_TM")
    spectrum.rename_result("T_ss", "T_ss_TM")
    spectrum.rename_result("time_elapsed", "time_elapsed_TM")
    print("TM linear basis done.")
    spectrum.calculate_refl_trans(circ=True, method="TM")
    spectrum.rename_result("R_RR", "R_RR_TM")
    spectrum.rename_result("R_RL", "R_RL_TM")
    spectrum.rename_result("R_LR", "R_LR_TM")
    spectrum.rename_result("R_LL", "R_LL_TM")
    spectrum.rename_result("T_RR", "T_RR_TM")
    spectrum.rename_result("T_RL", "T_RL_TM")
    spectrum.rename_result("T_LR", "T_LR_TM")
    spectrum.rename_result("T_LL", "T_LL_TM")
    print("TM circular basis done.")
    spectrum.calculate_refl_trans(circ=False, method="EM")
    spectrum.rename_result("R_pp", "R_pp_EM")
    spectrum.rename_result("R_ps", "R_ps_EM")
    spectrum.rename_result("R_sp", "R_sp_EM")
    spectrum.rename_result("R_ss", "R_ss_EM")
    spectrum.rename_result("T_pp", "T_pp_EM")
    spectrum.rename_result("T_ps", "T_ps_EM")
    spectrum.rename_result("T_sp", "T_sp_EM")
    spectrum.rename_result("T_ss", "T_ss_EM")
    spectrum.rename_result("time_elapsed", "time_elapsed_EM")
    print("EM linear basis done.")
    spectrum.calculate_refl_trans(circ=True, method="EM")
    spectrum.rename_result("R_RR", "R_RR_EM")
    spectrum.rename_result("R_RL", "R_RL_EM")
    spectrum.rename_result("R_LR", "R_LR_EM")
    spectrum.rename_result("R_LL", "R_LL_EM")
    spectrum.rename_result("T_RR", "T_RR_EM")
    spectrum.rename_result("T_RL", "T_RL_EM")
    spectrum.rename_result("T_LR", "T_LR_EM")
    spectrum.rename_result("T_LL", "T_LL_EM")
    print("EM circular basis done.")


def plot_all_spectra(spectrum, ax_R, ax_T):
    colors = pl.cm.Set2(np.linspace(0, 1, 8))
    data_refl_TM = 0.5 * (spectrum.data['R_RR_TM']
                          + spectrum.data['R_LL_TM']
                          + spectrum.data['R_RL_TM']
                          + spectrum.data['R_LR_TM'])
    data_refl_SM = 0.5 * (spectrum.data['R_RR_SM']
                          + spectrum.data['R_LL_SM']
                          + spectrum.data['R_RL_SM']
                          + spectrum.data['R_LR_SM'])
    data_refl_EM = 0.5 * (spectrum.data['R_RR_EM']
                          + spectrum.data['R_LL_EM']
                          + spectrum.data['R_RL_EM']
                          + spectrum.data['R_LR_EM'])

    data_trans_TM = 0.5 * (spectrum.data['T_RR_TM']
                           + spectrum.data['T_LL_TM']
                           + spectrum.data['T_RL_TM']
                           + spectrum.data['T_LR_TM'])
    data_trans_SM = 0.5 * (spectrum.data['T_RR_SM']
                           + spectrum.data['T_LL_SM']
                           + spectrum.data['T_RL_SM']
                           + spectrum.data['T_LR_SM'])
    data_trans_EM = 0.5 * (spectrum.data['T_RR_EM']
                           + spectrum.data['T_LL_EM']
                           + spectrum.data['T_RL_EM']
                           + spectrum.data['T_LR_EM'])

    ax_R.plot(wl_nm_list, data_refl_TM, color=colors[0], label='TM')
    ax_R.plot(wl_nm_list, data_refl_EM, color=colors[2], linestyle='dotted', label='EM')
    ax_R.plot(wl_nm_list, data_refl_SM, color=colors[1], linestyle='dashed', label='SM')

    ax_T.plot(wl_nm_list, data_trans_TM, color=colors[0], label='TM')
    ax_T.plot(wl_nm_list, data_trans_EM, color=colors[2], linestyle='dotted', label='EM')
    ax_T.plot(wl_nm_list, data_trans_SM, color=colors[1], linestyle='dashed', label='SM')

    plt.sca(ax_R)
    plt.ylim([0, 1.5])
    plt.xlim([400, 800])
    plt.legend(loc=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')

    plt.sca(ax_T)
    plt.ylim([0, 1.5])
    plt.xlim([400, 800])
    plt.legend(loc=3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance')

# Common parameters
n_av = 1.433
pitch_nm = 480
chole = ch.Cholesteric(pitch360=pitch_nm)
n_entry = n_av
n_exit = n_av
theta_in_deg = 0
theta_in_rad = theta_in_deg * np.pi / 180
wl_nm_list = np.arange(400, 800, 1)  # 0.2 for nicer spectral resolution

# Cholesteric 1
biref_1 = 0.05
n_e_1 = n_av + 0.5 * biref_1
n_o_1 = n_av - 0.5 * biref_1
N_per_1 = 20

# Cholesteric 2
biref_2 = 0.2
n_e_2 = n_av + 0.5 * biref_2
n_o_2 = n_av - 0.5 * biref_2
N_per_2 = 200

# Calculation of the spectrum
spectrum_1 = ll.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole, n_e=n_e_1, n_o=n_o_1, n_entry=n_entry,
                                                            n_exit=n_exit, N_per=N_per_1, theta_in_rad=theta_in_rad))
spectrum_2 = ll.Spectrum(wl_nm_list, "CholestericModel", dict(chole=chole, n_e=n_e_2, n_o=n_o_2, n_entry=n_entry,
                                                            n_exit=n_exit, N_per=N_per_2, theta_in_rad=theta_in_rad))
calculate_all_spectra(spectrum_1)
calculate_all_spectra(spectrum_2)

fig = plt.figure()
ax_00 = fig.add_subplot(221)
ax_01 = fig.add_subplot(222)
ax_10 = fig.add_subplot(223)
ax_11 = fig.add_subplot(224)

plot_all_spectra(spectrum_1, ax_00, ax_10)
plot_all_spectra(spectrum_2, ax_01, ax_11)

ax_00.set_title("Reflectance, small system")
ax_10.set_title("Transmittance, small system")
ax_01.set_title("Reflectance, big system")
ax_11.set_title("Transmittance, big system")

plt.tight_layout()

fig.savefig('script_spectra_SM_TM_EM_figure.png', dpi=300)

print('Time elapsed for the small system:')
print('SM: ' + str(spectrum_1.data['time_elapsed_SM']))
print('TM: ' + str(spectrum_1.data['time_elapsed_TM']))
print('EM: ' + str(spectrum_1.data['time_elapsed_EM']))

print('Time elapsed for the big system:')
print('SM: ' + str(spectrum_2.data['time_elapsed_SM']))
print('TM: ' + str(spectrum_2.data['time_elapsed_TM']))
print('EM: ' + str(spectrum_2.data['time_elapsed_EM']))

plt.show()


