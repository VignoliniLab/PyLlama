import pyllama as ll
import numpy as np


"""
This script prints a few Layer/Structure parameters from SlabModel.
"""

# Parameters of the slab
n_av = 1.5
biref = 0.5
n_e = n_av + 0.5 * biref
n_o = n_av - 0.5 * biref
eps0 = np.array([[n_e**2, 0, 0],
                 [0, n_o**2, 0],
                 [0, 0, n_o**2]])
rotangle_deg = 30
rotangle_rad = rotangle_deg * np.pi / 180
rotaxis = 'z'

n_entry = 1
n_exit = 1.5
thickness_nm = 500
theta_in_deg = 30
theta_in_rad = theta_in_deg * np.pi / 180
wl = 500

# Creation of the Model
model = ll.SlabModel(eps0, thickness_nm, n_entry, n_exit, wl, theta_in_rad, rotangle_rad, rotaxis)

# Access to the transition matrix
print('P (transition matrix)')
print(model.structure.entry.P)
print(model.structure.layers[0].P)
print(model.structure.exit.P)

# Access to the propagation matrix
print('Q (propagation matrix)')
print(model.structure.entry.Q)
print(model.structure.layers[0].Q)
print(model.structure.exit.Q)

# Access to the eigenvectors
print('Eigenvectors')
print(model.structure.entry.eigenvectors)
print(model.structure.layers[0].eigenvectors)
print(model.structure.exit.eigenvectors)

# Access to the eigenvalues
print('Eigenvalues')
print(model.structure.entry.eigenvalues)
print(model.structure.layers[0].eigenvalues)
print(model.structure.exit.eigenvalues)

# Access to the Fresnel coefficients
print("Fresnel coefficients, reflection")
print(model.structure._get_fresnel_SM()[0])
print(model.structure._get_fresnel_TM()[0])
print(model.structure._get_fresnel_EM()[0])
print("Fresnel coefficients, transmission")
print(model.structure._get_fresnel_SM()[1])
print(model.structure._get_fresnel_TM()[1])
print(model.structure._get_fresnel_EM()[1])

# Access to the reflectance and transmittance
print('Reflectance')
print(model.get_refl_trans(method="SM")[0])
print(model.get_refl_trans(method="TM")[0])
print(model.get_refl_trans(method="EM")[0])
print('Transmittance')
print(model.get_refl_trans(method="SM")[1])
print(model.get_refl_trans(method="TM")[1])
print(model.get_refl_trans(method="EM")[1])

