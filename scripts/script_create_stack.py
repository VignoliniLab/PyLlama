import pyllama as pl
import numpy as np
import matplotlib.pyplot as plt


"""
This script consists in 3 sections that create a stack with Layer/Structure, with Model and with Spectrum
to check that the results are the same.
The user can choose the parameters in the first section.
"""


###########################
#                         #
#    COMMON PARAMETERS    #
#                         #
###########################

fig = plt.figure()
ax = fig.add_subplot(111)

# Entry and exit half-spaces
n_entry = 1.2
epsilon_entry = np.array([[n_entry ** 2, 0, 0],
                          [0, n_entry ** 2, 0],
                          [0, 0, n_entry ** 2]])
n_exit = 1.3
epsilon_exit = np.array([[n_exit ** 2, 0, 0],
                         [0, n_exit ** 2, 0],
                         [0, 0, n_exit ** 2]])

# Two layers in the periodic pattern
eps_a = np.array([[1.8 ** 2, 0, 0],
                   [0, 1.2 ** 2, 0],
                   [0, 0, 1.2 ** 2]])
eps_b = np.array([[2.1 ** 2, 0, 0],
                  [0, 1.8 ** 2, 0],
                  [0, 0, 1.8 ** 2]])
rot_a_rad = 40 * np.pi / 180
rot_b_rad = 10 * np.pi / 180
eps_a = pl.Layer.rotate_permittivity(eps_a, rot_a_rad, 'z')
eps_b = pl.Layer.rotate_permittivity(eps_b, rot_b_rad, 'z')

thick_a = 200  # in nm
thick_b = 300  # in nm

# Number of periods
N = 100

# Incident angle
theta_in_rad = 0

# Wavelengths
wl_nm_list = range(400, 800)


########################
#                      #
#    WITH STRUCTURE    #
#                      #
########################

# Creation of an empty variable
reflection_with_structure = []

# Calculation of the reflectance for each wavelength
print("Computation with Structure starting.")
for wl_nm in wl_nm_list:
    # Calculation of the wavevector
    k0 = 2 * np.pi / wl_nm
    Kx = n_entry * np.sin(theta_in_rad)
    Ky = 0
    Kz_entry = n_entry * np.cos(theta_in_rad)
    theta_out = np.arcsin((n_entry / n_exit) * np.sin(theta_in_rad))
    Kz_exit = n_exit * np.cos(theta_out)

    # Creation of the entry and exit half-spaces and of the two layers
    entry = pl.HalfSpace(epsilon_entry, Kx, Kz_entry, k0)
    exit = pl.HalfSpace(epsilon_exit, Kx, Kz_exit, k0)
    layer_a = pl.Layer(eps_a, thick_a, Kx, k0)
    layer_b = pl.Layer(eps_b, thick_b, Kx, k0)

    # Creation of the periodic stack
    my_stack_structure = pl.Structure(entry, exit, Kx, Ky, Kz_entry, Kz_exit, k0)
    my_stack_structure.add_layers([layer_a, layer_b])
    my_stack_structure.N_periods = N

    # Calculation of the reflectance and storage
    J_lin, _ = my_stack_structure.get_refl_trans()
    reflection_with_structure.append(J_lin[0, 1])

    print("Wavelength %s done." % (str(wl_nm)))

# Plotting
plt.plot(wl_nm_list, reflection_with_structure, label="with Structure")


####################
#                  #
#    WITH MODEL    #
#                  #
####################

# Creation of an empty variable
reflection_with_model = []

# Calculation of the reflectance for each wavelength
print("Computation with Model starting.")
for wl_nm in wl_nm_list:
    # Creation of the periodic stack
    my_stack_model = pl.StackModel([eps_a, eps_b],
                                    [thick_a, thick_b],
                                    n_entry,
                                    n_exit,
                                    wl_nm,
                                    theta_in_rad,
                                    N)

    # Calculation of the reflectance and storage
    J_lin, _ = my_stack_model.get_refl_trans()
    reflection_with_model.append(J_lin[0, 1])

    print("Wavelength %s done." % (str(wl_nm)))

# Plotting
plt.plot(wl_nm_list, reflection_with_model, linestyle="dashed", label="with Model")


#######################
#                     #
#    WITH SPECTRUM    #
#                     #
#######################

# Parameters for the stack
model_type = "StackModel"
model_parameters = {"eps_list": [eps_a, eps_b],
                    "thickness_nm_list": [thick_a, thick_b],
                    "n_entry": n_entry,
                    "n_exit": n_exit,
                    "theta_in_rad": theta_in_rad,
                    "N_per": N}

# Creation of the periodic stack
my_stack_spec = pl.Spectrum(wl_nm_list, model_type, model_parameters)

# Calculation of the reflectance spectrum in one go
print("Computation with Spectrum starting.")
my_stack_spec.calculate_refl_trans(talk=True)
reflection_with_spectrum = my_stack_spec.data["R_ps"]

# Plotting
plt.plot(wl_nm_list, reflection_with_spectrum, linestyle="dotted", label="with Spectrum")

plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')

# Saving the figure
#fig.savefig('script_create_stack_figure.png', dpi=300)

plt.show()


