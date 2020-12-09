import numpy as np
from numpy import linalg as la_np
from scipy import linalg as la_sp
import warnings
import time  # to calculate the time elapsed for a spectrum
import pickle  # to save as Pickle file
import scipy.io  # to save as Matlab file

# np.set_printoptions(precision=4) # TODO remove

# Constants
c = 2.998e8  # speed of light in vacuum

# vec_u = unit vector
x_u = np.array([1, 0, 0])
y_u = np.array([0, 1, 0])
z_u = np.array([0, 0, 1])

# Threshold for zero
thr = 1e-7


# General functions
def rot_mat(axis=np.array([0, 0, 1]), theta_rad=0):
    """
    This function builds a rotation matrix for a given angle around a given axis accordung to
    https://en.wikipedia.org/wiki/Rotation_matrix.

    :param ndarray axis: rotation axis: a one-dimensional Numpy array of length 3 (or the string ``'x'``, ``'y'`` or ``'z'``)
    :param float theta_rad: rotation angle (in radians) around the rotation axis ``axis``
    :return: a 3x3 Numpy array rotation matrix
    """
    # Axis as char to make it easier to read
    if isinstance(axis, str):
        if axis == 'x':
            axis = np.array([1, 0, 0])
        elif axis == 'y':
            axis = np.array([0, 1, 0])
        elif axis == 'z':
            axis = np.array([0, 0, 1])
        else:
            raise Exception('Invalid rotation axis.')
    if la_np.norm(axis) == 0:
        raise Exception('Invalid axis. Axis can not be (0, 0, 0).')
    else:
        # theta_rad = theta_rad % (2 * np.pi)
        axis = axis / la_np.norm(axis)
        ux = axis[0]
        uy = axis[1]
        uz = axis[2]
        costheta = np.cos(theta_rad)
        sintheta = np.sin(theta_rad)
        return np.array([[costheta + (ux ** 2) * (1 - costheta), ux * uy * (1 - costheta) - uz * sintheta,
                          ux * uz * (1 - costheta) + uy * sintheta],
                         [uy * ux * (1 - costheta) + uz * sintheta, costheta + (uy ** 2) * (1 - costheta),
                          uy * uz * (1 - costheta) - ux * sintheta],
                         [uz * ux * (1 - costheta) - uy * sintheta, uz * uy * (1 - costheta) + ux * sintheta,
                          costheta + (uz ** 2) * (1 - costheta)]])


class Wave(object):
    """
    This class represents a partial wave in a layer of the multilayer stack with:

    - its electric field
    - its magnetic field
    - its Poynting vector
    - the :math:`x`- (tangential) component of its normalised wavevector

    :param ndarray epsilon: permittivity tensor: a 3x3 Numpy array
    :param float Kx: :math:`x`-component of the normalised wavevector
    :param float Ex: :math:`x`-component of the electric field
    :param float Ey: :math:`y`-component of the electric field
    :param float Hx: :math:`x`-component of the magnetic field
    :param float Hy: :math:`y`-component of the magnetic field
    """

    def __init__(self, epsilon, Kx, Ex, Ey, Hx, Hy):
        Ez, Hz = Wave.calc_Ez_Hz(epsilon, Kx, Ex, Ey, Hy)
        # self.wavevector = [Kx, Ky, Kz_entry]
        # self.k0 = k0
        self.Kx = Kx
        self.elec = [Ex, Ey, Ez]
        self.magnet = [Hx, Hy, Hz]
        S = self.calc_poynting()
        self.poynting = S

    @staticmethod
    def _calc_cp(x, y):
        deno = (np.abs(x) ** 2 + np.abs(y) ** 2)
        if deno == 0:
            return 0
        else:
            return np.abs(x) ** 2 / deno  # (np.abs(x) ** 2 + np.abs(y) ** 2)

    def calc_cp_poynting(self):
        """
        This function calculates the parameter Cp, used to sort a pair of partial waves between s-like and p-like
        polarisations.
        :return: Cp = |Sx| ** 2 / (|Sx|**2 + |Sy|**2)
        """
        return Wave._calc_cp(self.poynting[0], self.poynting[1])

    def calc_cp_elec(self):
        """
        This function calculates the parameter Cp, used to sort a pair of partial waves between s and p
        polarisations.
        :return: Cp = |Ex| ** 2 / (|Ex|**2 + |Ey|**2)
        """
        return Wave._calc_cp(self.elec[0], self.elec[1])

    @staticmethod
    def calc_Ez_Hz(epsilon, Kx, Ex, Ey, Hy):
        """
        This function calculates the :math:`z`-components of the electric and magnetic fields of the Wave
        """
        Ez = - (epsilon[2, 0] / epsilon[2, 2]) * Ex - (epsilon[2, 1] / epsilon[2, 2]) * Ey - (Kx / epsilon[2, 2]) * Hy
        Hz = Kx * Ey
        return Ez, Hz

    def calc_poynting(self):
        """
        This function calculates the Poynting vector of the Wave
        """
        Sx = self.elec[1] * self.magnet[2] - self.elec[2] * self.magnet[1]
        Sy = self.elec[2] * self.magnet[0] - self.elec[0] * self.magnet[2]
        Sz = self.elec[0] * self.magnet[1] - self.elec[1] * self.magnet[0]
        # print("-----")
        # print("E:")
        # print(self.elec)
        # print("H:")
        # print(self.magnet)
        # print("S:")
        # print([Sx, Sy, Sz])
        # print("-----")
        return [Sx, Sy, Sz]

    @staticmethod
    def waves_to_matrix(w_list, norm=False):
        r"""
        Given a layer’s 4 partial waves as ``Waves`` objects, this function returns a matrix where the components
        :math:`E_x`, :math:`E_y`, :math:`H_x` and :math:`H_y` are arranged so that the matrix can be used as a
        transition matrix for the layer. The function extracts a vector :math:`\psi = [E_x, H_y, E_y, -H_x]` for each
        of the 4 waves and formats them into a matrix where each column is a :math:`\psi`.

        :param list w_list: list of 4 partial waves ``Waves`` ``[w_0, w_1, w_2, w_3]``
        :param bool norm: set to True to normalise each :math:`\psi` to a modulus of 1, otherwive set to False. Must
                     be set to False for the partial waves of the ``HalfSpaces``.

        :return: 4x4 Numpy array of 4 eigenvectors :math:`\psi_0`, :math:`\psi_1`, :math:`\psi_2` and :math:`\psi_3` whose values correspond to:

                 .. math::
                     \begin{bmatrix}
                         E_{x, \: 0} & E_{x, \: 1} & E_{x, \: 2} & E_{x, \: 3} \\
                         H_{y, \: 0} & H_{y, \: 1} & H_{y, \: 2} & H_{y, \: 3} \\
                         E_{y, \: 0} & E_{y, \: 1} & E_{y, \: 2} & E_{y, \: 3} \\
                         -H_{x, \: 0} & -H_{x, \: 1} & -H_{x, \: 2} & -H_{x, \: 3}
                     \end{bmatrix}
        """
        if len(w_list) != 4:
            raise Exception('List of four Waves required as parameter.')
        else:
            # TODO give the possibility to normalise!
            return np.array([
                [w_list[0].elec[0], w_list[1].elec[0], w_list[2].elec[0], w_list[3].elec[0]],
                [w_list[0].magnet[1], w_list[1].magnet[1], w_list[2].magnet[1], w_list[3].magnet[1]],
                [w_list[0].elec[1], w_list[1].elec[1], w_list[2].elec[1], w_list[3].elec[1]],
                [-w_list[0].magnet[0], -w_list[1].magnet[0], -w_list[2].magnet[0], -w_list[3].magnet[0]]
            ])

    @staticmethod
    def matrix_to_waves(mat, epsilon, Kx):
        r"""
        Given a layer’s 4 eigenvectors in a 4x4 Numpy array where each column corresponds to a column vector
        :math:`\psi = [E_x, H_y, E_y, -H_x]`, this function returns a list of the 4 corresponding Waves, which are
        easier to manipulate when electric fields, magnetic fields and Poynting vectors need to be accessed.

        :param ndarray mat: 4x4 Numpy array of 4 eigenvectors :math:`\psi_0`, :math:`\psi_1`, :math:`\psi_2` and :math:`\psi_3`
                            whose values correspond to:

                            .. math::
                                \begin{bmatrix}
                                    E_{x, \: 0} & E_{x, \: 1} & E_{x, \: 2} & E_{x, \: 3} \\
                                    H_{y, \: 0} & H_{y, \: 1} & H_{y, \: 2} & H_{y, \: 3} \\
                                    E_{y, \: 0} & E_{y, \: 1} & E_{y, \: 2} & E_{y, \: 3} \\
                                    -H_{x, \: 0} & -H_{x, \: 1} & -H_{x, \: 2} & -H_{x, \: 3}
                                \end{bmatrix}

        :param ndarray epsilon: permittivity tensor: 3x3 Numpy array
        :param float Kx: the :math:`x`- (tangential) component of its normalised wavevector
        :return: list of 4 corresponding ``Waves`` ``[w_0, w_1, w_2, w_3]``
        """
        if mat.shape != (4, 4):
            raise Exception('4x4 matrix required as parameter.')
        else:
            return [Wave(epsilon, Kx, mat[0, k], mat[2, k], -mat[3, k], mat[1, k]) for k in range(0, 4, 1)]


class Layer(object):
    """
    This class represents a homogeneous layer in a multilayer stack and enables to build Berreman’s matrix
    as well as the partial waves (eigenvalues, eigenvectors) of the layer. The layer is made of a
    non-magnetic and non-optically acvive material. ``Layer`` represents the physical layer for one specific
    wavelength (the material may be dispersive). Its parameters are:

    :param ndarray epsilon: permittivity tensor, a 3x3 Numpy array
    :param float thickness_nm: thickness of the ``Layer`` in nanometers
    :param float Kx: :math:`x`-component of the normalised wavevector
    :param float k0: normalisation factor of the wavevector: the :math:`x`-component of the wavevector is equal to
                     :math:`k_x = k_0 K_x`
    :param float rot_angle_rad: rotation angle of the layer (in radians) around the axis `rot_axis`
    :param ndarray rot_axis: rotation axis: a one-dimensional Numpy array of length 3 (or the string ``'x'``, ``'y'`` or ``'z'``)
    :param bool hold: when the user decides to hold (``hold=True``) the calculation of Berreman’s matrix, the eigenvalues
        and eigenvectors, the user must then manually apply the functions to the ``Layer`` before calculating the
        transfer or scattering matrix. This is exceptional practice. The default is ``hold=True``.
    :param String numerical_method: indicates the package to use to calculate the eigenvectors and eigenvalues of
        the layer; either ``'numpy'`` (default) or ``'sympy'``
    """

    def __init__(self, epsilon, thickness_nm, Kx, k0, rot_angle_rad=0, rot_axis='z', hold=False,
                 numerical_method="numpy"):
        # Assumption: not magnetic, mu = 1
        # Assumption: not optically active, rho = 0, rho' = 0
        if rot_angle_rad != 0:
            epsilon = Layer.rotate_permittivity(epsilon, rot_angle_rad, rot_axis)
        self.eps = epsilon
        self.thickness = thickness_nm
        self.Kx = Kx
        self.k0 = k0
        self.M = self._build_M()
        self.A = self._build_A()
        self.D = self._build_D()
        if hold:
            self.eigenvectors = None
            self.eigenvalues = None
            self.partial_waves = None
            self.P = None
            self.Q = None
        else:
            p_sorted, q_sorted, partial_waves_sorted = self._calc_p_q_sorted(numerical_method=numerical_method)
            self.eigenvectors = p_sorted
            self.eigenvalues = q_sorted
            self.partial_waves = partial_waves_sorted
            Pmat, Qmat = self.build_P_Q()
            self.P = Pmat
            self.Q = Qmat

    @staticmethod
    def rotate_permittivity(eps, angle_rad, axis='z'):
        """
        This function calculates a rotated permittivity tensor.

        :param ndarray eps: permittivity tensor: a 3x3 Numpy array
        :param float angle_rad: rotation angle (in radians) around the rotation axis ``axis``
        :param ndarray axis: rotation axis: a one-dimensional Numpy array of length 3 (or the string ``'x'``, ``'y'`` or ``'z'``)
        :return: rotated permittivity tensor: a 3x3 Numpy array
        """
        # Retrieve the rotation axis
        if isinstance(axis, str):
            if axis == 'x':
                axis = x_u
            elif axis == 'y':
                axis = y_u
            elif axis == 'z':
                axis = z_u
            else:
                raise Exception('Invalid rotation axis.')
        else:
            raise Exception('Invalid rotation axis.')
        # Rotation matrix
        R = rot_mat(axis, angle_rad)
        # Rotate permittivity tensor
        eps_rot = la_np.multi_dot((R, eps, R.transpose()))
        return eps_rot

    def _build_M(self):
        r"""
        This function calculates the matrix that describes the optical properties of the material.

        :return: :math:`M` matrix: 6x6 Numpy array
                 .. math::
                             \begin{bmatrix}
                                 \epsilon_{xx} & \epsilon_{xy} & \epsilon_{xz} & \rho_{xx} & \rho_{xy} & \rho_{xz} \\
                                 \epsilon_{yx} & \epsilon_{yy} & \epsilon_{yz} & \rho_{yx} & \rho_{yy} & \rho_{yz} \\
                                 \epsilon_{zx} & \epsilon_{zy} & \epsilon_{zz} & \rho_{zx} & \rho_{zy} & \rho_{zz} \\
                                 \rho'_{xx} & \rho'_{xy} & \rho'_{xz} & \mu_{xx} & \mu_{xy} & \mu_{xz} \\
                                 \rho'_{yx} & \rho'_{yy} & \rho'_{yz} & \mu_{yx} & \mu_{yy} & \mu_{yz} \\
                                 \rho'_{zx} & \rho'_{zy} & \rho'_{zz} & \mu_{zx} & \mu_{zy} & \mu_{zz} \\
                             \end{bmatrix}
        """
        return np.array([
            [self.eps[0, 0], self.eps[0, 1], self.eps[0, 2], 0, 0, 0],
            [self.eps[1, 0], self.eps[1, 1], self.eps[1, 2], 0, 0, 0],
            [self.eps[2, 0], self.eps[2, 1], self.eps[2, 2], 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

    def _build_A(self):
        """
        This function calculates the :math:`A` matrix from Passler (DOI 10.1364/JOSAB.34.002128). It is an intermediary
        matrix to calculate Berreman’s matrix.

        :return: :math:`A` matrix: 3x3 Numpy array
        """
        M = self.M
        b = M[2, 2] * M[5, 5] - M[2, 5] * M[5, 2]

        A20 = (M[5, 0] * M[2, 5] - M[2, 0] * M[5, 5]) / b
        A21 = ((M[5, 1] - self.Kx) * M[2, 5] - M[2, 1] * M[5, 5]) / b
        A22 = 0
        A23 = (M[5, 3] * M[2, 5] - M[2, 3] * M[5, 5]) / b
        A24 = (M[5, 4] * M[2, 5] - (M[2, 4] + self.Kx) * M[5, 5]) / b
        A25 = 0

        A50 = (M[5, 2] * M[2, 0] - M[2, 2] * M[5, 0]) / b
        A51 = (M[5, 2] * M[2, 1] - M[2, 2] * (M[5, 1] - self.Kx)) / b
        A52 = 0
        A53 = (M[5, 2] * M[2, 3] - M[2, 2] * M[5, 3]) / b
        A54 = (M[5, 2] * (M[2, 4] + self.Kx) - M[2, 2] * M[5, 4]) / b
        A55 = 0

        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [A20, A21, A22, A23, A24, A25],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [A50, A51, A52, A53, A54, A55]
        ])

    def _build_D(self):
        """
        This function calculates Berreman’s matrix :math:`\Delta` (DOI 10.1364/JOSA.62.000502]

        :return: :math:`\Delta` matrix: 3x3 Numpy array
        """
        # Delta matrix from Berreman with pre-removal of zeros
        return np.array([
            [- self.Kx * self.eps[2, 0] / self.eps[2, 2],
             1 - self.Kx ** 2 / self.eps[2, 2],
             - self.Kx * self.eps[2, 1] / self.eps[2, 2],
             0],
            [self.eps[0, 0] - (self.eps[0, 2] * self.eps[2, 0]) / self.eps[2, 2],
             - self.Kx * self.eps[0, 2] / self.eps[2, 2],
             self.eps[0, 1] - (self.eps[0, 2] * self.eps[2, 1]) / self.eps[2, 2],
             0],
            [0, 0, 0, 1],
            [self.eps[1, 0] - (self.eps[1, 2] * self.eps[2, 0]) / self.eps[2, 2],
             - self.Kx * self.eps[1, 2] / self.eps[2, 2],
             - self.Kx ** 2 + self.eps[1, 1] - (self.eps[1, 2] * self.eps[2, 1]) / self.eps[2, 2],
             0]
        ])

    def _calc_p_q_unsorted(self, numerical_method="numpy"):
        r"""
        This function calculates the ``Layer``’s eigenvectors and eigenvalues.

        :param str method: ``"numpy"`` or ``"sympy"``: enables to choose the calculation method:

                - ``"numpy"``: numerical calculation with the Numpy package
                - ``"sympy"``: symbolic calculation and evaluation with the Sympy package

        :return p_unsorted: 4x4 array of eigenvectors whose values correspond to:

                .. math::
                    \begin{bmatrix}
                        E_{x, \: 0} & E_{x, \: 1} & E_{x, \: 2} & E_{x, \: 3} \\
                        H_{y, \: 0} & H_{y, \: 1} & H_{y, \: 2} & H_{y, \: 3} \\
                        E_{y, \: 0} & E_{y, \: 1} & E_{y, \: 2} & E_{y, \: 3} \\
                        -H_{x, \: 0} & -H_{x, \: 1} & -H_{x, \: 2} & -H_{x, \: 3}
                    \end{bmatrix}

        :return q_unsorted: 4x4 diagonal matrix of eigenvalues
        :return partial_waves_unsorted: list of ``Waves`` corresponding to the eigenvectors ``p_unsorted``
        """
        if numerical_method == "sympy":
            return self._calc_p_q_unsorted_sympy()
        elif numerical_method == "numpy":
            return self._calc_p_q_unsorted_numpy()

    def _calc_p_q_unsorted_sympy(self, precision=20):
        """
        This function calculates the ``Layer``’s eigenvectors and eigenvalues with the Sympy package (symbolic
        calculation and evaluation).
        :param int precision: precision at which to evaluate the eigenvalues and eigenvectors
        :return p_unsorted, q_unsorted, partial_waves_unsorted: matrices of eigenvectors (in columns), eigenvalues
                (diagonal matrix) and list of corresponding ``Waves``
        """
        import sympy as sy
        # Convert Numpy matrix to Sympy matrix
        D_sympy = sy.Matrix(self.D)
        # Calculate the eigenvalues and the non-normalised eigenvectors
        res_sympy = D_sympy.eigenvects()
        # Normalise the eigenvectors
        res0 = res_sympy[0][2][0] / res_sympy[0][2][0].norm()
        res1 = res_sympy[1][2][0] / res_sympy[1][2][0].norm()
        res2 = res_sympy[2][2][0] / res_sympy[2][2][0].norm()
        res3 = res_sympy[3][2][0] / res_sympy[3][2][0].norm()
        # Cast the eigenvalues and eigenvectors in the appropriate variables
        p0_sy = np.array(res0.evalf(precision))
        p1_sy = np.array(res1.evalf(precision))
        p2_sy = np.array(res2.evalf(precision))
        p3_sy = np.array(res3.evalf(precision))
        q0_sy = complex(np.array(res_sympy[0][0].evalf(precision)))
        q1_sy = complex(np.array(res_sympy[1][0].evalf(precision)))
        q2_sy = complex(np.array(res_sympy[2][0].evalf(precision)))
        q3_sy = complex(np.array(res_sympy[3][0].evalf(precision)))
        # p_unsorted = np.stack((p0_sy, p1_sy, p2_sy, p3_sy), axis=1)
        # p_unsorted = p_unsorted.astype(float)
        p0_sy = p0_sy.flatten()
        p1_sy = p1_sy.flatten()
        p2_sy = p2_sy.flatten()
        p3_sy = p3_sy.flatten()
        p_unsorted = np.array([[p0_sy[0], p1_sy[0], p2_sy[0], p3_sy[0]],
                               [p0_sy[1], p1_sy[1], p2_sy[1], p3_sy[1]],
                               [p0_sy[2], p1_sy[2], p2_sy[2], p3_sy[2]],
                               [p0_sy[3], p1_sy[3], p2_sy[3], p3_sy[3]]])
        p_unsorted = p_unsorted.astype(complex)
        partial_waves_unsorted = Wave.matrix_to_waves(p_unsorted, self.eps, self.Kx)
        q_unsorted = np.array([q0_sy, q1_sy, q2_sy, q3_sy])
        return p_unsorted, q_unsorted, partial_waves_unsorted

    def _calc_p_q_unsorted_numpy(self):
        """
        This function calculates the ``Layer``’s eigenvectors and eigenvalues with the Numpy package (numerical
        calculation).
        :return p_unsorted, q_unsorted, partial_waves_unsorted: matrices of eigenvectors (in columns), eigenvalues
                (diagonal matrix) and list of corresponding ``Waves``
        """
        q_unsorted, p_unsorted = la_np.eig(self.D)
        partial_waves_unsorted = Wave.matrix_to_waves(p_unsorted, self.eps, self.Kx)
        return p_unsorted, q_unsorted, partial_waves_unsorted

    def _sort_p_q(self, p_unsorted, q_unsorted, partial_waves_unsorted):
        r"""
        This function sorts the ``Layer``’s eigenvectors and eigenvalues.

        :param ndarray p_unsorted: 4x4 array of eigenvectors whose values correspond to:

                                    .. math::
                                        \begin{bmatrix}
                                            E_{x, \: 0} & E_{x, \: 1} & E_{x, \: 2} & E_{x, \: 3} \\
                                            H_{y, \: 0} & H_{y, \: 1} & H_{y, \: 2} & H_{y, \: 3} \\
                                            E_{y, \: 0} & E_{y, \: 1} & E_{y, \: 2} & E_{y, \: 3} \\
                                            -H_{x, \: 0} & -H_{x, \: 1} & -H_{x, \: 2} & -H_{x, \: 3}
                                        \end{bmatrix}

        :param ndarray q_unsorted: 4x4 diagonal matrix of eigenvalues
        :param list partial_waves_unsorted: list of ``Waves`` corresponding to the eigenvectors ``p_unsorted``

        :return p_sorted: 4x4 array of eigenvectors whose columns are the same as ``p_unsorted``’s but sorted
        :return q_sorted: 4x4 diagonal matrix of eigenvalues whose columns are the same as ``q_unsorted``’s but sorted
        :return partial_waves_sorted: list of 4 ``Waves`` corresponding to the eigenvectors ``p_sorted``
        """
        # Determine the reflection and transmission eigs
        # Analysing with Kz (sometimes fails):
        """
        for k in range(0, 4):
            q_k = q_unsorted[k]
            if np.isreal(q_k):
                test_variable = np.real(q_k)
            else:
                test_variable = np.imag(q_k)
            if test_variable >= 0:
                id_trans.append(k)
            else:
                id_refl.append(k)
        """
        # Analyse with the Poynting vector:
        id_refl = []
        id_trans = []
        for k in range(0, 4, 1):
            S_z_k = partial_waves_unsorted[k].poynting[2]
            if np.isreal(S_z_k):
                test_variable = np.real(S_z_k)
            else:
                test_variable = np.imag(S_z_k)
            if test_variable > 0:
                id_trans.append(k)
            else:
                id_refl.append(k)

        # Sort in a unique way by analysing p
        # Find out whether it is birefringent
        # Calculate Cp0 and Cp1 using Pointing vectors
        Cp0 = partial_waves_unsorted[id_trans[0]].calc_cp_poynting()
        Cp1 = partial_waves_unsorted[id_trans[1]].calc_cp_poynting()
        if np.abs(Cp0 - Cp1) > thr:
            # It is birefringent
            # Sort the trans
            if Cp1 < Cp0:
                id_trans = [id_trans[1], id_trans[0]]
            Cp0 = partial_waves_unsorted[id_refl[0]].calc_cp_poynting()
            Cp1 = partial_waves_unsorted[id_refl[1]].calc_cp_poynting()
            # Sort the refl
            if Cp1 < Cp0:
                id_refl = [id_refl[1], id_refl[0]]
        else:
            # It is not birefringent
            # Sort the trans
            Cp0 = partial_waves_unsorted[id_trans[0]].calc_cp_elec()
            Cp1 = partial_waves_unsorted[id_trans[1]].calc_cp_elec()
            if (Cp1 - Cp0) < thr:
                id_trans = [id_trans[1], id_trans[0]]
            # Sort the refl
            Cp0 = partial_waves_unsorted[id_refl[0]].calc_cp_elec()
            Cp1 = partial_waves_unsorted[id_refl[1]].calc_cp_elec()
            if (Cp1 - Cp0) < thr:
                id_refl = [id_refl[1], id_refl[0]]

        # Sort p and q
        order = [id_trans[1], id_trans[0], id_refl[1], id_refl[0]]

        q_sorted = np.array([q_unsorted[order[0]], q_unsorted[order[1]], q_unsorted[order[2]], q_unsorted[order[3]]])

        p_sorted = np.stack((p_unsorted[:, order[0]].transpose(), p_unsorted[:, order[1]].transpose(),
                             p_unsorted[:, order[2]].transpose(), p_unsorted[:, order[3]].transpose()), axis=1)
        partial_waves_sorted = [partial_waves_unsorted[order[0]], partial_waves_unsorted[order[1]],
                                partial_waves_unsorted[order[2]], partial_waves_unsorted[order[3]]]
        return p_sorted, q_sorted, partial_waves_sorted

    def _correct_p(self, q_sorted):
        """
        Passler’s equations (DOI 10.1364/JOSAB.34.002128), not used here.
        """
        # Make gamma
        gamma00 = 1
        gamma11 = 1
        gamma31 = 1
        gamma20 = -1
        if np.abs(q_sorted[0] - q_sorted[1]) < thr:
            gamma01 = 0
            gamma02 = - (self.eps[2, 0] + self.Kx * q_sorted[0]) / (self.eps[2, 2] - self.Kx ** 2)
            gamma10 = 0
            gamma12 = - (self.eps[2, 1]) / (self.eps[2, 2] - self.Kx ** 2)
        else:
            gamma01 = (self.eps[1, 2] * (self.eps[2, 0] + self.Kx * q_sorted[0]) - self.eps[1, 0] * (
                    self.eps[2, 2] - self.Kx ** 2)) \
                      / ((self.eps[2, 2] - self.Kx ** 2) * (self.eps[1, 1] - self.Kx ** 2 - q_sorted[0] ** 2) -
                         self.eps[1, 2] * self.eps[2, 1])
            gamma02 = - ((self.eps[2, 0] + self.Kx * q_sorted[0]) / (self.eps[2, 2] - self.Kx ** 2)) \
                      - gamma01 * ((self.eps[2, 1]) / (self.eps[2, 2] - self.Kx ** 2))
            gamma10 = (self.eps[2, 1] * (self.eps[0, 2] + self.Kx * q_sorted[1]) - self.eps[0, 1] * (
                    self.eps[2, 2] - self.Kx ** 2)) \
                      / ((self.eps[2, 2] - self.Kx ** 2) * (self.eps[0, 0] - q_sorted[1] ** 2) - (
                    self.eps[0, 2] + self.Kx * q_sorted[1]) * (self.eps[2, 0] + self.Kx * q_sorted[1]))
            gamma12 = - gamma10 * ((self.eps[2, 0] + self.Kx * q_sorted[1]) / (self.eps[2, 2] - self.Kx ** 2)) \
                      - ((self.eps[2, 1]) / (self.eps[2, 2] - self.Kx ** 2))
        if np.abs(q_sorted[2] - q_sorted[3]) <= thr:
            gamma21 = 0
            gamma22 = (self.eps[2, 0] + self.Kx * q_sorted[2]) / (self.eps[2, 2] - self.Kx ** 2)
            gamma30 = 0
            gamma32 = - (self.eps[2, 1]) / (self.eps[2, 2] - self.Kx ** 2)
        else:
            gamma21 = ((self.eps[1, 0] * (self.eps[2, 2] + self.Kx ** 2)) - (
                    self.eps[1, 2] * (self.eps[2, 0] + self.Kx * q_sorted[2]))) \
                      / (((self.eps[2, 2] + self.Kx ** 2) * (self.eps[1, 1] - self.Kx ** 2 - q_sorted[2] ** 2)) - (
                    self.eps[1, 2] * self.eps[2, 1]))
            gamma22 = ((self.eps[2, 0] + self.Kx * q_sorted[2]) / (self.eps[2, 2] - self.Kx ** 2)) \
                      + gamma21 * ((self.eps[2, 1]) / (self.eps[2, 2] - self.Kx ** 2))
            gamma30 = (self.eps[2, 1] * (self.eps[0, 2] + self.Kx * q_sorted[3]) - self.eps[0, 1] * (
                    self.eps[2, 2] - self.Kx ** 2)) \
                      / (((self.eps[2, 2] - self.Kx ** 2) * (self.eps[0, 0] - q_sorted[3] ** 2)) - (
                    (self.eps[0, 2] + self.Kx * q_sorted[3]) * (self.eps[2, 0] + self.Kx * q_sorted[3])))
            gamma32 = - gamma30 * ((self.eps[2, 0] + self.Kx * q_sorted[3]) / (self.eps[2, 2] - self.Kx ** 2)) \
                      - (self.eps[2, 1] / (self.eps[2, 2] - self.Kx ** 2))

        # Trick if it's nan (see Passler)
        # /!\ in Python, 0/0 = not a number, number/0 = sign(number)*Inf
        if np.isnan(gamma01) or np.isinf(gamma01) or gamma01 > 1e6:
            gamma01 = 0
        if np.isnan(gamma02) or np.isinf(gamma02) or gamma02 > 1e6:
            gamma02 = - ((self.eps[2, 0] + self.Kx * q_sorted[0]) / (self.eps[2, 2] - self.Kx ** 2))
        if np.isnan(gamma10) or np.isinf(gamma10) or gamma10 > 1e6:
            gamma10 = 0
        if np.isnan(gamma12) or np.isinf(gamma12) or gamma12 > 1e6:
            gamma12 = - ((self.eps[2, 1]) / (self.eps[2, 2] - self.Kx ** 2))
        if np.isnan(gamma21) or np.isinf(gamma21) or gamma21 > 1e6:
            gamma21 = 0
        if np.isnan(gamma22) or np.isinf(gamma22) or gamma22 > 1e6:
            gamma22 = ((self.eps[2, 0] + self.Kx * q_sorted[2]) / (self.eps[2, 2] - self.Kx ** 2))
        if np.isnan(gamma30) or np.isinf(gamma30) or gamma30 > 1e6:
            gamma30 = 0
        if np.isnan(gamma32) or np.isinf(gamma32) or gamma32 > 1e6:
            gamma32 = - (self.eps[2, 1] / (self.eps[2, 2] - self.Kx ** 2))
        p_corrected = np.array([
            [gamma00, gamma10, gamma20, gamma30],
            [q_sorted[0] * gamma00 - self.Kx * gamma02,
             q_sorted[1] * gamma10 - self.Kx * gamma12,
             q_sorted[2] * gamma20 - self.Kx * gamma22,
             q_sorted[3] * gamma30 - self.Kx * gamma32],
            [gamma01, gamma11, gamma21, gamma31],
            [q_sorted[0] * gamma01,
             q_sorted[1] * gamma11,
             q_sorted[2] * gamma21,
             q_sorted[3] * gamma31]
        ])
        partial_waves_corrected = Wave.matrix_to_waves(p_corrected, self.eps, self.Kx)
        return p_corrected, partial_waves_corrected

    def _calc_p_q_sorted(self, numerical_method="numpy"):
        r"""
        This function calculates the ``Layer``’s eigenvectors and eigenvalues and returns them sorted according to their
        drection of propagation and their polarisation.

        :return ndarray p_sorted: 4x4 array of eigenvectors whose values correspond to:

                                    .. math::
                                        \begin{bmatrix}
                                            E_{x, \: 0} & E_{x, \: 1} & E_{x, \: 2} & E_{x, \: 3} \\
                                            H_{y, \: 0} & H_{y, \: 1} & H_{y, \: 2} & H_{y, \: 3} \\
                                            E_{y, \: 0} & E_{y, \: 1} & E_{y, \: 2} & E_{y, \: 3} \\
                                            -H_{x, \: 0} & -H_{x, \: 1} & -H_{x, \: 2} & -H_{x, \: 3}
                                        \end{bmatrix}

        :return ndarray q_sorted: 4x4 diagonal matrix of eigenvalues
        :return list partial_waves_sorted: list of 4 ``Waves`` corresponding to the eigenvectors ``p_sorted``
        """
        # Calculate unsorted
        p_unsorted, q_unsorted, partial_waves_unsorted = self._calc_p_q_unsorted(numerical_method=numerical_method)
        # Sort them
        p_sorted, q_sorted, partial_waves_sorted = self._sort_p_q(p_unsorted, q_unsorted, partial_waves_unsorted)
        # p_sorted, partial_waves_sorted = self._correct_p(q_sorted)  # not used here, but can be un-commented
        return p_sorted, q_sorted, partial_waves_sorted

    def build_P_Q(self):
        """
        This function constructs the interface matrix :math:`P` and the propagation matrix :math:`Q` for one ``Layer``.

            - The interface matrix :math:`P` describes the change of medium.
            - The propagation matrix :math:`Q` describes the propagation in the thickness of the medium and the phase
              build-up.

        :return ndarray P: interface matrix :math:`P`, 3x3 Numpy array
        :return ndarray Q: propagation matrix :math:`Q`, 3x3 Numpy array
        """
        P = self.eigenvectors
        Q = np.array([[np.exp(1j * self.k0 * self.eigenvalues[0] * self.thickness), 0, 0, 0],
                      [0, np.exp(1j * self.k0 * self.eigenvalues[1] * self.thickness), 0, 0],
                      [0, 0, np.exp(1j * self.k0 * self.eigenvalues[2] * self.thickness), 0],
                      [0, 0, 0, np.exp(1j * self.k0 * self.eigenvalues[3] * self.thickness)]])
        return P, Q


class HalfSpace(Layer):
    """
    This class represents an isotropic semi-infinite medium before or after a multilayer stack and enables to build
    the partial waves (eigenvalues, eigenvectors) of the medium. ``HalfSpace`` represents the physical layer for one
    specific wavelength (the material may be dispersive).

    :param ndarray epsilon: permittivity tensor: a 3x3 Numpy array
    :param float Kx: :math:`x`-component of the normalised wavevector
    :param float k0: normalisation factor of the wavevector: the :math:`x`-component of the wavevector is equal to
               :math:`k_x = k_0 K_x`
    :param str category: always ``"isotropic"`` in the code’s 1.0 version
    """

    # __doc__ += Layer.__doc__
    def __init__(self, epsilon, Kx, Kz, k0, category="isotropic"):
        if category == "isotropic":
            if epsilon[0, 0] == epsilon[1, 1] == epsilon[2, 2]:
                if all(v == 0 for v in
                       [epsilon[0, 1], epsilon[0, 2], epsilon[1, 0], epsilon[1, 2], epsilon[2, 0], epsilon[2, 1]]):
                    valid = True
                else:
                    valid = False
            else:
                valid = False
        else:
            raise Exception("Invalid category for the HalfSpace.")
        if valid:
            self.category = category
            thickness_nm = 0
            self.Kz = Kz
            super().__init__(epsilon, thickness_nm, Kx, k0)
        else:
            raise Exception("The permittivity of the HalfSpace does not correspond to its category.")

    def _calc_p_q_sorted(self, numerical_method="analytical"):
        """
        This function calculates the ``HalfSpace``’s eigenvectors and eigenvalues analytically and returns them sorted.

        :return p_sorted: 4x4 array of eigenvectors whose columns are the same as ``p_unsorted``’s but sorted
        :return q_sorted: 4x4 diagonal matrix of eigenvalues whose columns are the same as ``q_unsorted``’s but sorted
        :return partial_waves_sorted: list of 4 ``Waves`` corresponding to the eigenvectors ``p_sorted``
        """
        if self.category == "isotropic":
            # Get the index from Kx
            n = np.sqrt(self.eps[0, 0])
            # Get the angle from Kx
            # sqrt gives real number (or nan), unless the argument is complex
            sin_phi = self.Kx / n
            cos_phi = np.sqrt(1 - sin_phi ** 2 + 0j)
            q_sorted = [n * cos_phi, n * cos_phi, -n * cos_phi, -n * cos_phi]
            p_sorted_mat = np.array([
                [cos_phi, 0, cos_phi, 0],
                [n, 0, -n, 0],
                [0, 1, 0, 1],
                [0, n * cos_phi, 0, -n * cos_phi]
            ])
            partial_waves_sorted = Wave.matrix_to_waves(p_sorted_mat, self.eps, self.Kx)
            return p_sorted_mat, q_sorted, partial_waves_sorted
        else:
            raise Exception('Invalid HalfSpace category.')


class Structure(object):
    r"""
    This class represents a multilayer stack by:

    - a list of layers (instances of ``Layer``), initially an empty list
    - an isotropic entry semi-infinite medium (instance of ``HalfSpace``)
    - an isotropic exit semi-infinite medium (instance of ``HalfSpace``)
    - the number of periods :math:`N`, which means that the list of layers will be repeated :math:`N` times

    :param float Kx: :math:`x`-component of the normalised wavevector (stays the same throughought the stack)
    :param float Ky: :math:`y`-component of the normalised wavevector (equal to 0 by construction)
    :param float Kz: :math:`z`-component of the normalised wavevector (changes in each layer)
    :param float k0: normalisation factor of the wavevector:

                     .. math::
                         \begin{bmatrix}
                             k_x \\
                             k_y \\
                             k_z
                         \end{bmatrix}
                         =
                         k_0
                         \begin{bmatrix}
                             K_x \\
                             K_y \\
                             K_z
                         \end{bmatrix}

                     which stays the same throughout the stack and depends on the wavelength

    :param int N_periods: the number of periods

    """

    def __init__(self, entry, exit, Kx, Ky, Kz_entry, Kz_exit, k0, N_periods=1):
        self.layers = []  # contains a list of objects Layer
        self.Kx = Kx  # kx = Kx * k0
        self.Ky = Ky  # ky = Ky * k0
        self.Kz_entry = Kz_entry  # kz_entry = Kz_entry * k0
        self.Kz_exit = Kz_exit  # kz_exit = Kz_exit * k0
        self.k0 = k0
        if Structure.is_layer_compatible(self, entry):
            self.entry = entry  # contains a layer object for isotropic medium
        else:
            raise Exception(
                "The entry half-space doesn't have the same Kx and k0 as the structure, it cannot be added.")
        if Structure.is_layer_compatible(self, exit):
            self.exit = exit  # contains a layer object for isotropic medium
        else:
            raise Exception("The exit half-space doesn't have the same Kx and k0 as the structure, it cannot be added.")
        if not isinstance(N_periods, int) and (int(N_periods) != N_periods):
            raise Exception('The number of periods must be an integer.')
        if N_periods < 0:
            raise Exception('The number of periods must be positive.')
        self.N_periods = N_periods  # number of repeats

    def is_layer_compatible(self, layer):
        """
        This function checks if the layer ``layer`` is compatible with the structure: the :math:`x`-component of the
        normalised wavevector and its normalisation factor :math:`k_0` must stay the same throughout the stack.

        :param layer: a ``Layer``
        :return bool: ``True`` if the layer is compatible with the structure, ``False`` otherwise
        """
        if (layer.Kx == self.Kx) and (layer.k0 == self.k0):
            return True
        else:
            return False

    @staticmethod
    def are_structures_compatible(structures_list):
        """
        This function checks if several structures are compatible with each other: the :math:`x`-component of the
        normalised wavevector and its normalisation factor :math:`k_0` must stay the same in all structures.

        :param list structures_list: a list of ``Structures``
        :return bool: ``True`` if all ``Structures`` from the list are compatible, ``False`` otherwise
        """
        # TODO test this function
        result = True
        sk = 0
        while result and sk < len(structures_list):
            result = ((structures_list[sk].Kx == structures_list[0].Kx) and (
                        structures_list[sk].k0 == structures_list[0].k0))
            sk = sk + 1
        return result

    def add_layer(self, new_layer):
        """
        This function adds a ``Layer`` to the structure, provided it is compatible with the structure (see function
        ``Structure.is_layer_compatible(layer)``).

        :param new_layer: a ``Layer`` to add
        """
        if self.is_layer_compatible(new_layer):
            self.layers.append(new_layer)
        else:
            raise Exception("The new layer doesn't have the same Kx and k0 as the structure, it cannot be added.")

    def add_layers(self, new_layers_list):
        """
        This function adds ``Layers`` to the structure, provided they are compatible with the structure (see function
        ``Structure.is_layer_compatible(layer)``). Compatible ``Layers`` are added, non-compatible ``Layers`` are not
        added.

        :param new_layers_list: a list of ``Layers`` to add
        """
        compatible_layers = []
        klay = 0
        for lay in new_layers_list:
            if self.is_layer_compatible(lay):
                compatible_layers.append(lay)
            else:
                warnings.warn("The new layer number " + str(
                    klay) + " doesn't have the same Kx and k0 as the structure, it has not been added.")
            klay = klay + 1
        self.layers.extend(compatible_layers)
        return compatible_layers

    def remove_layer(self, layer_index):
        """
        This function removes from the ``Structure`` the ``Layer`` at the index ``layer_index``.

        :param int layer_index: index of the ``Layer`` to remove
        """
        self.layers.pop(layer_index)

    def replace_layer(self, layer_index, new_layer):
        """
        This function replaces the ``Layer`` at the index ``layer_index`` by the ``Layer`` ``new_layer`` provided it is
        compatible with the structure (see function ``Structure.is_layer_compatible(layer)``).

        :param int layer_index: index of the ``Layer`` to replace
        :param new_layer: ``Layer`` to add
        """
        if self.is_layer_compatible(new_layer):
            self.layers[layer_index] = new_layer
        else:
            raise Exception("The new layer doesn't have the same Kx and k0 as the structure, it cannot be added.")

    @staticmethod
    def _powers_of_two(n):
        """
        This function decomposes an integer in powers of two: ``bin(n)`` converts n from decimal to binary and browsing
        through all digits enables to take only these equal to 1 and enumerate their position (``enumerate`` returns
        (counter, value) tuples). When converting binary numbers to strings, Python adds "0b" in front of the string so
        the first two characters fo the string are not used (hence the ``[:1:-1]``).

        :param int n: integer to decompose in powers of two
        :return: list of powers of two
        """
        return [p for p, v in enumerate(bin(n)[:1:-1]) if int(v)]

    @staticmethod
    def build_scattering_matrix_to_next(layer_a, layer_b):
        """
        This function constructs the scattering matrix :math:`S_{ab}` between two successive layers :math:`a` and
        :math:`b` by taking into acount the following phenomena:

        - the propagation through the first layer with the propagation matrix Q of ``layer_a``
        - the transition from the first layer (``layer_a``’s matrix P) and the second layer (``layer_b``’s matrix P)

        :param ndarray layer_a: the first ``Layer``
        :param ndarray layer_b: the second ``Layer``
        :return: partial scattering matrix from layer :math:`a` to layer :math:`b`, a 4x4 Numpy array
        """
        Q_forward = np.array([
            [layer_a.Q[0, 0], layer_a.Q[0, 1], 0, 0],
            [layer_a.Q[1, 0], layer_a.Q[1, 1], 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        Q_backward = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, layer_a.Q[2, 2], layer_a.Q[2, 3]],
            [0, 0, layer_a.Q[3, 2], layer_a.Q[3, 3]]
        ])

        P_out = np.array([
            [layer_a.P[0, 0], layer_a.P[0, 1], -layer_b.P[0, 2], -layer_b.P[0, 3]],
            [layer_a.P[1, 0], layer_a.P[1, 1], -layer_b.P[1, 2], -layer_b.P[1, 3]],
            [layer_a.P[2, 0], layer_a.P[2, 1], -layer_b.P[2, 2], -layer_b.P[2, 3]],
            [layer_a.P[3, 0], layer_a.P[3, 1], -layer_b.P[3, 2], -layer_b.P[3, 3]]
        ])

        P_in = np.array([
            [layer_b.P[0, 0], layer_b.P[0, 1], -layer_a.P[0, 2], -layer_a.P[0, 3]],
            [layer_b.P[1, 0], layer_b.P[1, 1], -layer_a.P[1, 2], -layer_a.P[1, 3]],
            [layer_b.P[2, 0], layer_b.P[2, 1], -layer_a.P[2, 2], -layer_a.P[2, 3]],
            [layer_b.P[3, 0], layer_b.P[3, 1], -layer_a.P[3, 2], -layer_a.P[3, 3]]
        ])

        S12 = la_np.multi_dot((la_np.inv(Q_backward), la_np.inv(P_in), P_out, Q_forward))
        return S12

    @staticmethod
    def combine_scattering_matrices(S_ab, S_bc):
        """
        This function constructs the scattering matrix between three successive layers :math:`a`, :math:`b` and
        :math:`c` by combining the scattering matrices :math:`S_{ab}` from layer :math:`a` to layer :math:`b` and
        :math:`S_{bc}` from layer :math:`b` to layer :math:`c`.

        :param ndarray S_ab: the scattering matrix from layer :math:`a` to layer :math:`b`, a 4x4 Numpy array
        :param ndarray S_bc: the scattering matrix from layer :math:`b` to layer :math:`c`, a 4x4 Numpy array
        :return: partial scattering matrix from layer :math:`a` to layer :math:`c`, a 4x4 Numpy array
        """
        S_ab00 = np.array([
            [S_ab[0, 0], S_ab[0, 1]],
            [S_ab[1, 0], S_ab[1, 1]],
        ])
        S_ab01 = np.array([
            [S_ab[0, 2], S_ab[0, 3]],
            [S_ab[1, 2], S_ab[1, 3]],
        ])
        S_ab10 = np.array([
            [S_ab[2, 0], S_ab[2, 1]],
            [S_ab[3, 0], S_ab[3, 1]],
        ])
        S_ab11 = np.array([
            [S_ab[2, 2], S_ab[2, 3]],
            [S_ab[3, 2], S_ab[3, 3]],
        ])
        S_bc00 = np.array([
            [S_bc[0, 0], S_bc[0, 1]],
            [S_bc[1, 0], S_bc[1, 1]],
        ])
        S_bc01 = np.array([
            [S_bc[0, 2], S_bc[0, 3]],
            [S_bc[1, 2], S_bc[1, 3]],
        ])
        S_bc10 = np.array([
            [S_bc[2, 0], S_bc[2, 1]],
            [S_bc[3, 0], S_bc[3, 1]],
        ])
        S_bc11 = np.array([
            [S_bc[2, 2], S_bc[2, 3]],
            [S_bc[3, 2], S_bc[3, 3]],
        ])
        C = la_np.inv(np.identity(2) - np.dot(S_ab01, S_bc10))
        S_ac00 = la_np.multi_dot((S_bc00, C, S_ab00))
        S_ac01 = S_bc01 + la_np.multi_dot((S_bc00, C, S_ab01, S_bc11))
        S_ac10 = S_ab10 + la_np.multi_dot((S_ab11, S_bc10, C, S_ab00))
        S_ac11 = la_np.multi_dot((S_ab11, (np.identity(2) + la_np.multi_dot((S_bc10, C, S_ab01))), S_bc11))

        S_ac = np.array([
            [S_ac00[0, 0], S_ac00[0, 1], S_ac01[0, 0], S_ac01[0, 1]],
            [S_ac00[1, 0], S_ac00[1, 1], S_ac01[1, 0], S_ac01[1, 1]],
            [S_ac10[0, 0], S_ac10[0, 1], S_ac11[0, 0], S_ac11[0, 1]],
            [S_ac10[1, 0], S_ac10[1, 1], S_ac11[1, 0], S_ac11[1, 1]]
        ])
        return S_ac

    def build_exponential_matrix(self):
        """
        This function calculates the transfer matrix of the system with the exponential of Berreman’s matrix.

        :return: transfer matrix, a 4x4 Numpy array
        """
        E = self._build_exponential_matrix_partial()
        E = la_np.multi_dot((la_np.inv(self.exit.P), E, self.entry.P))
        return E

    def _build_exponential_matrix_partial(self):
        """
        This function calculates the partial transfer matrix of the system with the exponential of Berreman’s matrix.
        The partial transfer matrix is the transfer matrix of the stack’s ``Layers`` with the exclusion of the entry and
        exit ``HalfSpaces``.

        :return: transfer matrix, a 4x4 Numpy array
        """
        N_layers = len(self.layers)
        E_period = np.identity(4)
        E = np.identity(4)
        for kl in range(0, N_layers, 1):
            E_layer = la_sp.expm(1j * self.layers[kl].D * self.layers[kl].thickness * self.k0)
            E_period = np.dot(E_layer, E_period)
        for kp in range(0, self.N_periods, 1):
            E = np.dot(E_period, E)
        return E

    def build_transfer_matrix(self):
        """
        This function calculates the transfer matrix of the system with the matrices of eigenvalues and eigenvectors.

        :return: transfer matrix, a 4x4 Numpy array
        """
        T = self._build_transfer_matrix_partial()
        T = la_np.multi_dot((la_np.inv(self.exit.P), T, self.entry.P))
        return T

    def _build_transfer_matrix_partial(self):
        """
        This function calculates the partial transfer matrix of the system with the matrices of eigenvalues and
        eigenvectors. The partial transfer matrix is the transfer matrix of the stack’s ``Layers`` with the exclusion of
        the entry and exit ``HalfSpaces``.

        :return: transfer matrix, a 4x4 Numpy array
        """
        # Note: the commented lines help to verify
        N_layers = len(self.layers)
        T_period = np.identity(4)
        for kl in range(0, N_layers, 1):
            T_layer = la_np.multi_dot((self.layers[kl].P, self.layers[kl].Q, la_np.inv(self.layers[kl].P)))
            T_period = np.dot(T_layer, T_period)
        if self.N_periods > 2:
            # Combine the complete periodic motives, WITH POWERS OF 2
            # (the commented lines help check the powers of 2 decomposition)
            T = T_period.copy()
            decomp_powers = Structure._powers_of_two(self.N_periods - 1)
            decomp_matrices = []
            # test_powers = []
            decomp_matrices.append(T)
            # test_powers.append(0)
            for kp in range(1, decomp_powers[-1] + 1):
                T = np.dot(T, T)
                decomp_matrices.append(T)
                # test_powers.append(kp)
                # print(kp)
            T = decomp_matrices[0]
            # test = 2**0
            # print(decomp_powers)
            # print('---')
            for kp in decomp_powers[0:]:
                # T = np.dot(T, decomp_matrices[kp])
                T = np.dot(decomp_matrices[kp], T)
                # test = test + 2**kp
            # print(test)
        elif self.N_periods == 2:
            T = T_period.copy()
            # T = np.dot(T, T_period)
            T = np.dot(T_period, T)
        else:
            T = T_period.copy()

        return T

    def build_scattering_matrix(self):
        """
        This function calculates the scattering matrix of the system.

        :return: scattering matrix, a 4x4 Numpy array
        """
        S = self._build_scattering_matrix_partial()
        # Combine the entry and exit medias
        S_entry = Structure.build_scattering_matrix_to_next(self.entry, self.layers[0])
        S_exit = Structure.build_scattering_matrix_to_next(self.layers[-1], self.exit)
        S = self.combine_scattering_matrices(S, S_exit)
        S = self.combine_scattering_matrices(S_entry, S)
        return S

    def _build_scattering_matrix_partial(self):
        """
        This function calculates the partial scattering matrix of the system. The partial scattering matrix is the
        scattering matrix of the stack’s ``Layers`` with the exclusion of the entry and exit ``HalfSpaces``.

        :return: partial scattering matrix, a 4x4 Numpy array
        """
        N_layers = len(self.layers)
        S_last_period = np.identity(4)
        # For the last (uncomplete) periodic motive that leads to the exit half-space:
        # We combine all layers; the last one will transition to th exit half-space and will be added later
        # there are N_layers layers in the full periodic motive
        # the bottom layer is indexed N_layers - 1
        # the top layer is indexed 0
        # at kl = N_layers-2 we combine the index N_layers-1 and the index N_layers-2
        # at kl = 0 we combine the index 1 the index 0
        for kl in range(N_layers - 2, -1, -1):
            S_layer = Structure.build_scattering_matrix_to_next(self.layers[kl], self.layers[kl + 1])
            S_last_period = self.combine_scattering_matrices(S_layer, S_last_period)
        # For the complete periodic motive (that leads to another pitch):
        # We take the uncomplete periodic motive and add the last layer transitionning to the first layer
        S_layer = Structure.build_scattering_matrix_to_next(self.layers[N_layers - 1], self.layers[0])
        S_period = self.combine_scattering_matrices(S_layer, S_last_period)
        if self.N_periods > 2:
            # Combine the complete periodic motives, WITH POWERS OF 2
            # (the commented lines help check the powers of 2 decomposition)
            S = S_period.copy()
            decomp_powers = Structure._powers_of_two(self.N_periods - 2)
            decomp_matrices = []
            # test_powers = []
            decomp_matrices.append(S)
            # test_powers.append(0)
            for kp in range(1, decomp_powers[-1] + 1):
                S = self.combine_scattering_matrices(S, S)
                decomp_matrices.append(S)
                # test_powers.append(kp)
                # print(kp)
            S = decomp_matrices[0]
            # test = 2**0
            # print(decomp_powers)
            # print('---')
            for kp in decomp_powers[0:]:
                S = self.combine_scattering_matrices(decomp_matrices[kp], S)
                # test = test + 2**kp
            # print(test)
            # Combine the last (uncomplete) periodic motive
            S = self.combine_scattering_matrices(S_last_period, S)
        elif self.N_periods == 2:
            S = S_period.copy()
            S = self.combine_scattering_matrices(S_last_period, S)
        else:
            S = S_last_period.copy()
        return S

    @staticmethod
    def build_exponential_matrix_multi(struct_list, entry, exit):
        r"""
        This function calculates the transfer matrix with the direct exponential of Berreman’s matrix for a system made
        of sub-stacks.

        :return: transfer matrix, a 4x4 Numpy array
        """
        # TODO: rephrase the exceptions
        if len(struct_list) == 1:
            E = struct_list[0].build_exponential_matrix()
        else:
            if isinstance(entry, HalfSpace) and isinstance(exit, HalfSpace):
                if Structure.are_structures_compatible(struct_list):
                    if struct_list[0].is_layer_compatible(entry) and struct_list[0].is_layer_compatible(exit):
                        E = np.identity(4)
                        for k in range(len(struct_list)):
                            E_chunk = struct_list[k]._build_exponential_matrix_partial()
                            E = np.dot(E_chunk, E)
                        # Combine the entry and exit medias
                        E = la_np.multi_dot((la_np.inv(exit.P), E, entry.P))
                    else:
                        raise Exception("The entry and/or exit half-spaces are not compatible with the structures.")
                else:
                    raise Exception("The structures are not compatible.")
            else:
                raise Exception("The entry and/or exit inputs are not HalfSpaces.")
        return E

    @staticmethod
    def build_transfer_matrix_multi(struct_list, entry, exit):
        r"""
        This function calculates the transfer matrix with the eigenvectors and eigenvalues for a system made of
        sub-stacks.

        :return: transfer matrix, a 4x4 Numpy array
        """
        # TODO: rephrase the exceptions
        if len(struct_list) == 1:
            T = struct_list[0].build_transfer_matrix()
        else:
            if isinstance(entry, HalfSpace) and isinstance(exit, HalfSpace):
                if Structure.are_structures_compatible(struct_list):
                    if struct_list[0].is_layer_compatible(entry) and struct_list[0].is_layer_compatible(exit):
                        T = np.identity(4)
                        for k in range(len(struct_list)):
                            T_chunk = struct_list[k]._build_transfer_matrix_partial()
                            T = np.dot(T_chunk, T)
                        # Combine the entry and exit medias
                        T = la_np.multi_dot((la_np.inv(exit.P), T, entry.P))
                    else:
                        raise Exception("The entry and/or exit half-spaces are not compatible with the structures.")
                else:
                    raise Exception("The structures are not compatible.")
            else:
                raise Exception("The entry and/or exit inputs are not HalfSpaces.")
        return T

    @staticmethod
    def build_scattering_matrix_multi(struct_list, entry, exit):
        r"""
        This function calculates the scattering matrix for a system made of sub-stacks.

        :return: scattering matrix, a 4x4 Numpy array
        """
        if len(struct_list) == 1:
            S = struct_list[0].build_scattering_matrix()
        else:
            if isinstance(entry, HalfSpace) and isinstance(exit, HalfSpace):
                if Structure.are_structures_compatible(struct_list):
                    if struct_list[0].is_layer_compatible(entry) and struct_list[0].is_layer_compatible(exit):
                        S = np.identity(4)
                        for k in range(len(struct_list)):
                            if k == 0:  # this is the first chunck
                                after = struct_list[k + 1].layers[0]
                                S_chunk = struct_list[k]._build_scattering_matrix_partial()
                                S = Structure.combine_scattering_matrices(S, S_chunk)
                                S_after = Structure.build_scattering_matrix_to_next(struct_list[k].layers[-1], after)
                                S = Structure.combine_scattering_matrices(S, S_after)
                            elif k == len(struct_list) - 1:  # this is the last chunk
                                S_chunk = struct_list[k]._build_scattering_matrix_partial()
                                S = Structure.combine_scattering_matrices(S, S_chunk)
                            else:
                                after = struct_list[k + 1].layers[0]
                                S_chunck = struct_list[k]._build_scattering_matrix_partial()
                                S = Structure.combine_scattering_matrices(S, S_chunck)
                                S_after = Structure.build_scattering_matrix_to_next(struct_list[k].layers[-1], after)
                                S = Structure.combine_scattering_matrices(S, S_after)
                        # Combine the entry and exit medias
                        S_entry = Structure.build_scattering_matrix_to_next(entry, struct_list[0].layers[0])
                        S_exit = Structure.build_scattering_matrix_to_next(struct_list[-1].layers[-1], exit)
                        S = Structure.combine_scattering_matrices(S_entry, S)
                        S = Structure.combine_scattering_matrices(S, S_exit)
                    else:
                        raise Exception("The entry and/or exit half-spaces are not compatible with the structures.")
                else:
                    raise Exception("The structures are not compatible.")
            else:
                raise Exception("The entry and/or exit Layers are not HalfSpaces.")
        return S

    def get_refl_trans(self, circ=False, method="SM"):
        r"""
        This function calculates the ``Structure``’s reflectance in the linear or circular polarisation basis, with the
        method chosen by the user.

        :param bool circ: ``False`` to express results in the linear polarisation basis, ``True`` to express results in
                          the circular polarisation basis
        :param string method: the matrix method to use for the calculation:

                              - ``"SM"`` for the scattering matrix method
                              - ``"TM"`` for the transfer matrix method with the eigenvectors and eigenvalues
                              - ``"EM"`` for the transfer matrix method with the direct exponential of Berreman’s matrix

        :return: reflectance: 2x2 Numpy array whose values correspond to:

                     - in the linear polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 R_{p \: \text{to} \: p} & R_{s \: \text{to} \: p} \\
                                 R_{p \: \text{to} \: s} & R_{s \: \text{to} \: s}
                             \end{bmatrix}

                    - in the circular polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 R_{RCP \: \text{to} \: RCP} & R_{LCP \: \text{to} \: RCP} \\
                                 R_{RCP \: \text{to} \: LCP} & R_{LCP \: \text{to} \: LCP}
                             \end{bmatrix}


        :return: transmittance: 2x2 Numpy array whose values correspond to:

                     - in the linear polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 T_{p \: \text{to} \: p} & T_{s \: \text{to} \: p} \\
                                 T_{p \: \text{to} \: s} & T_{s \: \text{to} \: s}
                             \end{bmatrix}

                    - in the circular polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 T_{RCP \: \text{to} \: RCP} & T_{LCP \: \text{to} \: RCP} \\
                                 T_{RCP \: \text{to} \: LCP} & T_{LCP \: \text{to} \: LCP}
                             \end{bmatrix}

        """
        if method == "SM":
            reflectance, transmittance = self._get_refl_trans_SM(circ=circ)
        elif method == "TM":
            reflectance, transmittance = self._get_refl_trans_TM(circ=circ)
        elif method == "EM":
            reflectance, transmittance = self._get_refl_trans_EM(circ=circ)
        else:
            raise Exception('Invalid method (only SM, TM and EM allowed).')
        return reflectance, transmittance

    def _get_refl_trans_SM(self, circ=False):
        """
        This function calculates the ``Structure``’s reflectance and transmittance in the linear or circular polarisation basis, with the
        scattering matrix method. See ``Structure.get_refl_trans`` for more details.
        """
        J_refl, J_trans = self._get_fresnel_SM()
        if circ:
            J_refl, J_trans = Structure.fresnel_to_fresnel_circ(J_refl, J_trans)
        R = np.array([
            [np.abs(J_refl[0, 0]) ** 2, np.abs(J_refl[0, 1]) ** 2],
            [np.abs(J_refl[1, 0]) ** 2, np.abs(J_refl[1, 1]) ** 2]
        ])
        factor = self.Kz_exit / self.Kz_entry
        T = np.array([
            [factor * np.abs(J_trans[0, 0]) ** 2, factor * np.abs(J_trans[0, 1]) ** 2],
            [factor * np.abs(J_trans[1, 0]) ** 2, factor * np.abs(J_trans[1, 1]) ** 2]
        ])
        return R, T

    def _get_refl_trans_TM(self, circ=False):
        """
        This function calculates the ``Structure``’s reflectance and transmittance in the linear or circular polarisation basis, with the
        transfer matrix method with the eigenvectors and eigenvalues. See ``Structure.get_refl_trans`` for more details.
        """
        J_refl, J_trans = self._get_fresnel_TM()
        if circ:
            J_refl, J_trans = Structure.fresnel_to_fresnel_circ(J_refl, J_trans)
        R = np.array([
            [np.abs(J_refl[0, 0]) ** 2, np.abs(J_refl[0, 1]) ** 2],
            [np.abs(J_refl[1, 0]) ** 2, np.abs(J_refl[1, 1]) ** 2]
        ])
        factor = self.Kz_exit / self.Kz_entry
        T = np.array([
            [factor * np.abs(J_trans[0, 0]) ** 2, factor * np.abs(J_trans[0, 1]) ** 2],
            [factor * np.abs(J_trans[1, 0]) ** 2, factor * np.abs(J_trans[1, 1]) ** 2]
        ])
        return R, T

    def _get_refl_trans_EM(self, circ=False):
        """
        This function calculates the ``Structure``’s reflectance and transmittance in the linear or circular polarisation basis, with the
        transfer matrix method with the direct exponential of Berreman’s matrix. See ``Structure.get_refl_trans`` for more
        details.
        """
        J_refl, J_trans = self._get_fresnel_EM()
        if circ:
            J_refl, J_trans = Structure.fresnel_to_fresnel_circ(J_refl, J_trans)
        R = np.array([
            [np.abs(J_refl[0, 0]) ** 2, np.abs(J_refl[0, 1]) ** 2],
            [np.abs(J_refl[1, 0]) ** 2, np.abs(J_refl[1, 1]) ** 2]
        ])
        factor = self.Kz_exit / self.Kz_entry
        T = np.array([
            [factor * np.abs(J_trans[0, 0]) ** 2, factor * np.abs(J_trans[0, 1]) ** 2],
            [factor * np.abs(J_trans[1, 0]) ** 2, factor * np.abs(J_trans[1, 1]) ** 2]
        ])
        return R, T

    def get_fresnel(self, method="SM"):
        r"""
        This function calculates the ``Structure``’s reflection and transmission coefficients in the linear polarisation basis, with the
        method chosen by the user.

        :param string method: the matrix method to use for the calculation:

                              - ``"SM"`` for the scattering matrix method
                              - ``"TM"`` for the transfer matrix method with the eigenvectors and eigenvalues
                              - ``"EM"`` for the transfer matrix method with the direct exponential of Berreman’s matrix

        :return J_refl: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             r_{p \: \text{to} \: p} & r_{s \: \text{to} \: p} \\
                             r_{p \: \text{to} \: s} & r_{s \: \text{to} \: s}
                         \end{bmatrix}

        :return J_trans: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             t_{p \: \text{to} \: p} & t_{s \: \text{to} \: p} \\
                             t_{p \: \text{to} \: s} & t_{s \: \text{to} \: s}
                         \end{bmatrix}
        """
        if method == "SM":
            J_refl, J_trans = self._get_fresnel_SM()
        elif method == "TM":
            J_refl, J_trans = self._get_fresnel_TM()
        elif method == "EM":
            J_refl, J_trans = self._get_fresnel_EM()
        else:
            raise Exception('Invalid method (only SM, TM and EM allowed).')
        return J_refl, J_trans

    def _get_fresnel_SM(self):
        """
        This function calculates the ``Structure``’s reflection and transmission coefficients in the linear polarisation basis, with the scattering
        matrix method. See ``Structure.get_fresnel`` for more details.
        """
        SM = self.build_scattering_matrix()

        J_refl = np.array([
            [SM[2, 0], SM[2, 1]],
            [SM[3, 0], SM[3, 1]]
        ])

        J_trans = np.array([
            [SM[0, 0], SM[0, 1]],
            [SM[1, 0], SM[1, 1]]
        ])
        return J_refl, J_trans

    def _get_fresnel_TM(self):
        """
        This function calculates the ``Structure``’s reflection and transmission coefficients in the linear polarisation basis, with the transfer
        matrix method with the eigenvectors and eigenvalues. See ``Structure.get_fresnel`` for more details.
        """
        TM = self.build_transfer_matrix()

        deno = TM[2, 2] * TM[3, 3] - TM[3, 2] * TM[2, 3]
        r_p_to_p = (TM[3, 0] * TM[2, 3] - TM[2, 0] * TM[3, 3]) / deno
        r_p_to_s = (TM[2, 0] * TM[3, 2] - TM[3, 0] * TM[2, 2]) / deno
        r_s_to_p = (TM[3, 1] * TM[2, 3] - TM[2, 1] * TM[3, 3]) / deno
        r_s_to_s = (TM[2, 1] * TM[3, 2] - TM[3, 1] * TM[2, 2]) / deno
        t_p_to_p = TM[0, 0] + TM[0, 2] * r_p_to_p + TM[0, 3] * r_p_to_s
        t_p_to_s = TM[1, 0] + TM[1, 2] * r_p_to_p + TM[1, 3] * r_p_to_s
        t_s_to_p = TM[0, 1] + TM[0, 2] * r_s_to_p + TM[0, 3] * r_s_to_s
        t_s_to_s = TM[1, 1] + TM[1, 2] * r_s_to_p + TM[1, 3] * r_s_to_s

        J_refl = np.array([
            [r_p_to_p, r_s_to_p],
            [r_p_to_s, r_s_to_s]
        ])

        J_trans = np.array([
            [t_p_to_p, t_s_to_p],
            [t_p_to_s, t_s_to_s]
        ])
        return J_refl, J_trans

    def _get_fresnel_EM(self):
        """
        This function calculates the ``Structure``’s reflection and transmission coefficients in the linear polarisation basis, with the transfer
        matrix method with the direct exponential of Berreman’s matrix. See ``Structure.get_fresnel`` for more
        details.
        """
        EM = self.build_exponential_matrix()

        deno = EM[2, 2] * EM[3, 3] - EM[3, 2] * EM[2, 3]
        r_p_to_p = (EM[3, 0] * EM[2, 3] - EM[2, 0] * EM[3, 3]) / deno
        r_p_to_s = (EM[2, 0] * EM[3, 2] - EM[3, 0] * EM[2, 2]) / deno
        r_s_to_p = (EM[3, 1] * EM[2, 3] - EM[2, 1] * EM[3, 3]) / deno
        r_s_to_s = (EM[2, 1] * EM[3, 2] - EM[3, 1] * EM[2, 2]) / deno
        t_p_to_p = EM[0, 0] + EM[0, 2] * r_p_to_p + EM[0, 3] * r_p_to_s
        t_p_to_s = EM[1, 0] + EM[1, 2] * r_p_to_p + EM[1, 3] * r_p_to_s
        t_s_to_p = EM[0, 1] + EM[0, 2] * r_s_to_p + EM[0, 3] * r_s_to_s
        t_s_to_s = EM[1, 1] + EM[1, 2] * r_s_to_p + EM[1, 3] * r_s_to_s

        J_refl = np.array([
            [r_p_to_p, r_s_to_p],
            [r_p_to_s, r_s_to_s]
        ])

        J_trans = np.array([
            [t_p_to_p, t_s_to_p],
            [t_p_to_s, t_s_to_s]
        ])
        return J_refl, J_trans

    @staticmethod
    def fresnel_to_fresnel_circ(J_refl, J_trans):
        r"""
        This function converts reflection and transmission coefficients in the linear polarisation basis to reflection
        and transmission coefficients in the circular polarisation bases.

        :param ndarray J_refl: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             r_{p \: \text{to} \: p} & r_{s \: \text{to} \: p} \\
                             r_{p \: \text{to} \: s} & r_{s \: \text{to} \: s}
                         \end{bmatrix}

        :param ndarray J_trans: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             t_{p \: \text{to} \: p} & t_{s \: \text{to} \: p} \\
                             t_{p \: \text{to} \: s} & t_{s \: \text{to} \: s}
                         \end{bmatrix}

        :return J_refl_c: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             r_{RCP \: \text{to} \: RCP} & r_{LCP \: \text{to} \: RCP} \\
                             r_{RCP \: \text{to} \: LCP} & r_{LCP \: \text{to} \: LCP}
                         \end{bmatrix}

        :return J_trans_c: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             t_{RCP \: \text{to} \: RCP} & t_{LCP \: \text{to} \: RCP} \\
                             t_{RCP \: \text{to} \: LCP} & t_{LCP \: \text{to} \: LCP}
                         \end{bmatrix}
        """
        F = np.array([
            [1, 1],
            [-1j, 1j]
        ])
        B = np.array([
            [1, 1],
            [1j, -1j]
        ])
        J_refl_c = la_np.multi_dot((la_np.inv(B), J_refl, F))
        J_trans_c = la_np.multi_dot((la_np.inv(F), J_trans, F))
        return J_refl_c, J_trans_c

    @staticmethod
    def get_fresnel_multi(structures_list, entry, exit, method="SM"):
        r"""
        This function calculates reflection and transmission coefficients for a system made of a list of Structures in the linear
        polarisation basis, with the method chosen by the user.

        :param list structures_list: list of multiple ``Structures`` that constitute the stack. Their respective entry
                                     and exit ``HalfSpaces`` will be ignored.
        :param entry: instance of ``HalfSpace`` that constitutes the stack’s entry semi-infinite medium
        :param exit: instance of ``HalfSpace`` that constitutes the stack’s exit semi-infinite medium
        :param string method: the matrix method to use for the calculation:

                              - ``"SM"`` for the scattering matrix method
                              - ``"TM"`` for the transfer matrix method with the eigenvectors and eigenvalues
                              - ``"EM"`` for the transfer matrix method with the direct exponential of Berreman’s matrix

        :return J_refl: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             r_{p \: \text{to} \: p} & r_{s \: \text{to} \: p} \\
                             r_{p \: \text{to} \: s} & r_{s \: \text{to} \: s}
                         \end{bmatrix}

        :return J_trans: 2x2 Numpy array whose values correspond to:

                     .. math::
                         \begin{bmatrix}
                             t_{p \: \text{to} \: p} & t_{s \: \text{to} \: p} \\
                             t_{p \: \text{to} \: s} & t_{s \: \text{to} \: s}
                         \end{bmatrix}
        """
        if method == "SM":
            J_refl, J_trans = Structure._get_fresnel_SM_multi(structures_list, entry, exit)
        elif method == "TM":
            J_refl, J_trans = Structure._get_fresnel_TM_multi(structures_list, entry, exit)
        elif method == "EM":
            J_refl, J_trans = Structure._get_fresnel_EM_multi(structures_list, entry, exit)
        else:
            raise Exception('Invalid method (only SM, TM and EM allowed).')
        return J_refl, J_trans

    @staticmethod
    def _get_fresnel_SM_multi(structures_list, entry, exit):
        """
        This function calculates the reflection and transmission coefficients for a system made of a list of Structures in the linear
        polarisation basis, with the scattering matrix method. See ``Structure.get_fresnel`` for more details.
        """
        SM = Structure.build_scattering_matrix_multi(structures_list, entry, exit)
        J_refl = np.array([
            [SM[2, 0], SM[2, 1]],
            [SM[3, 0], SM[3, 1]]
        ])

        J_trans = np.array([
            [SM[0, 0], SM[0, 1]],
            [SM[1, 0], SM[1, 1]]
        ])
        return J_refl, J_trans

    @staticmethod
    def _get_fresnel_TM_multi(structures_list, entry, exit):
        """
        This function calculates the reflection and transmission coefficients for a system made of a list of Structures in the linear
        polarisation basis, with the transfer matrix method with the eigenvectors and eigenvalues. See
        ``Structure.get_fresnel`` for more details.
        """
        TM = Structure.build_transfer_matrix_multi(structures_list, entry, exit)
        deno = TM[2, 2] * TM[3, 3] - TM[3, 2] * TM[2, 3]
        r_p_to_p = (TM[3, 0] * TM[2, 3] - TM[2, 0] * TM[3, 3]) / deno
        r_p_to_s = (TM[2, 0] * TM[3, 2] - TM[3, 0] * TM[2, 2]) / deno
        r_s_to_p = (TM[3, 1] * TM[2, 3] - TM[2, 1] * TM[3, 3]) / deno
        r_s_to_s = (TM[2, 1] * TM[3, 2] - TM[3, 1] * TM[2, 2]) / deno
        t_p_to_p = TM[0, 0] + TM[0, 2] * r_p_to_p + TM[0, 3] * r_p_to_s
        t_p_to_s = TM[1, 0] + TM[1, 2] * r_p_to_p + TM[1, 3] * r_p_to_s
        t_s_to_p = TM[0, 1] + TM[0, 2] * r_s_to_p + TM[0, 3] * r_s_to_s
        t_s_to_s = TM[1, 1] + TM[1, 2] * r_s_to_p + TM[1, 3] * r_s_to_s

        J_refl = np.array([
            [r_p_to_p, r_s_to_p],
            [r_p_to_s, r_s_to_s]
        ])

        J_trans = np.array([
            [t_p_to_p, t_s_to_p],
            [t_p_to_s, t_s_to_s]
        ])
        return J_refl, J_trans

    @staticmethod
    def _get_fresnel_EM_multi(structures_list, entry, exit):
        """
        This function calculates the reflection and transmission coefficients for a system made of a list of Structures in the linear
        polarisation basis, with the transfer matrix method with the direct exponential of Berreman’s matrix. See
        ``Structure.get_fresnel`` for more details.
        """
        EM = Structure.build_exponential_matrix_multi(structures_list, entry, exit)
        deno = EM[2, 2] * EM[3, 3] - EM[3, 2] * EM[2, 3]
        r_p_to_p = (EM[3, 0] * EM[2, 3] - EM[2, 0] * EM[3, 3]) / deno
        r_p_to_s = (EM[2, 0] * EM[3, 2] - EM[3, 0] * EM[2, 2]) / deno
        r_s_to_p = (EM[3, 1] * EM[2, 3] - EM[2, 1] * EM[3, 3]) / deno
        r_s_to_s = (EM[2, 1] * EM[3, 2] - EM[3, 1] * EM[2, 2]) / deno
        t_p_to_p = EM[0, 0] + EM[0, 2] * r_p_to_p + EM[0, 3] * r_p_to_s
        t_p_to_s = EM[1, 0] + EM[1, 2] * r_p_to_p + EM[1, 3] * r_p_to_s
        t_s_to_p = EM[0, 1] + EM[0, 2] * r_s_to_p + EM[0, 3] * r_s_to_s
        t_s_to_s = EM[1, 1] + EM[1, 2] * r_s_to_p + EM[1, 3] * r_s_to_s

        J_refl = np.array([
            [r_p_to_p, r_s_to_p],
            [r_p_to_s, r_s_to_s]
        ])

        J_trans = np.array([
            [t_p_to_p, t_s_to_p],
            [t_p_to_s, t_s_to_s]
        ])
        return J_refl, J_trans

    @staticmethod
    def get_refl_trans_multi(structures_list, entry, exit, circ=False, method="SM"):
        r"""
        This function calculates reflectance of a system made of a list of Structures in the linear or circular
        polarisation basis, with the method chosen by the user.

        :param list structures_list: list of multiple ``Structures`` that constitute the stack. Their respective entry
                                     and exit ``HalfSpaces`` will be ignored.
        :param entry: instance of ``HalfSpace`` that constitutes the stack’s entry semi-infinite medium
        :param exit: instance of ``HalfSpace`` that constitutes the stack’s exit semi-infinite medium
        :param bool circ: ``False`` to express results in the linear polarisation basis, ``True`` to express results in
                          the circular polarisation basis
        :param string method: the matrix method to use for the calculation:

                              - ``"SM"`` for the scattering matrix method
                              - ``"TM"`` for the transfer matrix method with the eigenvectors and eigenvalues
                              - ``"EM"`` for the transfer matrix method with the direct exponential of Berreman’s matrix

        :return: reflectance: 2x2 Numpy array whose values correspond to:

                     - in the linear polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 R_{p \: \text{to} \: p} & R_{s \: \text{to} \: p} \\
                                 R_{p \: \text{to} \: s} & R_{s \: \text{to} \: s}
                             \end{bmatrix}

                    - in the circular polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 R_{RCP \: \text{to} \: RCP} & R_{LCP \: \text{to} \: RCP} \\
                                 R_{RCP \: \text{to} \: LCP} & R_{LCP \: \text{to} \: LCP}
                             \end{bmatrix}


        :return: transmittance: 2x2 Numpy array whose values correspond to:

                     - in the linear polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 T_{p \: \text{to} \: p} & T_{s \: \text{to} \: p} \\
                                 T_{p \: \text{to} \: s} & T_{s \: \text{to} \: s}
                             \end{bmatrix}

                    - in the circular polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 T_{RCP \: \text{to} \: RCP} & T_{LCP \: \text{to} \: RCP} \\
                                 T_{RCP \: \text{to} \: LCP} & T_{LCP \: \text{to} \: LCP}
                             \end{bmatrix}
        """
        if method == "SM":
            reflectance, transmittance = Structure._get_refl_trans_multi_SM(structures_list, entry, exit, circ)
        elif method == "TM":
            reflectance, transmittance = Structure._get_refl_trans_multi_TM(structures_list, entry, exit, circ)
        elif method == "EM":
            reflectance, transmittance = Structure._get_refl_trans_multi_EM(structures_list, entry, exit, circ)
        else:
            raise Exception('Invalid method (only SM, TM and EM allowed).')
        return reflectance, transmittance

    @staticmethod
    def _get_refl_trans_multi_SM(structures_list, entry, exit, circ=False):
        """
        This function calculates the reflectance and transmittance of a system made of a list of Structures in the linear or circular
        polarisation basis with the scattering matrix method. See ``Structure.get_refl_trans_multi`` for more details.
        """
        J_refl, J_trans = Structure._get_fresnel_SM_multi(structures_list, entry, exit)
        if circ:
            J_refl, J_trans = Structure.fresnel_to_fresnel_circ(J_refl, J_trans)
        R = np.array([
            [np.abs(J_refl[0, 0]) ** 2, np.abs(J_refl[0, 1]) ** 2],
            [np.abs(J_refl[1, 0]) ** 2, np.abs(J_refl[1, 1]) ** 2]
        ])
        factor = exit.Kz / entry.Kz
        T = np.array([
            [factor * np.abs(J_trans[0, 0]) ** 2, factor * np.abs(J_trans[0, 1]) ** 2],
            [factor * np.abs(J_trans[1, 0]) ** 2, factor * np.abs(J_trans[1, 1]) ** 2]
        ])
        return R, T

    @staticmethod
    def _get_refl_trans_multi_TM(structures_list, entry, exit, circ=False):
        """
        This function calculates the reflectance and transmittance of a system made of a list of Structures in the linear or circular
        polarisation basis with the eigenvectors and eigenvalues. See ``Structure.get_refl_trans_multi`` for more details.
        """
        J_refl, J_trans = Structure._get_fresnel_TM_multi(structures_list, entry, exit)
        if circ:
            J_refl, J_trans = Structure.fresnel_to_fresnel_circ(J_refl, J_trans)
        R = np.array([
            [np.abs(J_refl[0, 0]) ** 2, np.abs(J_refl[0, 1]) ** 2],
            [np.abs(J_refl[1, 0]) ** 2, np.abs(J_refl[1, 1]) ** 2]
        ])
        factor = exit.Kz / entry.Kz
        T = np.array([
            [factor * np.abs(J_trans[0, 0]) ** 2, factor * np.abs(J_trans[0, 1]) ** 2],
            [factor * np.abs(J_trans[1, 0]) ** 2, factor * np.abs(J_trans[1, 1]) ** 2]
        ])
        return R, T

    @staticmethod
    def _get_refl_trans_multi_EM(structures_list, entry, exit, circ=False):
        """
        This function calculates the reflectance and transmittance of a system made of a list of Structures in the linear or circular
        polarisation basis with the transfer matrix method with the direct exponential of Berreman’s matrix. See
        ``Structure.get_refl_trans_multi`` for more details.
        """
        J_refl, J_trans = Structure._get_fresnel_EM_multi(structures_list, entry, exit)
        if circ:
            J_refl, J_trans = Structure.fresnel_to_fresnel_circ(J_refl, J_trans)
        R = np.array([
            [np.abs(J_refl[0, 0]) ** 2, np.abs(J_refl[0, 1]) ** 2],
            [np.abs(J_refl[1, 0]) ** 2, np.abs(J_refl[1, 1]) ** 2]
        ])
        factor = exit.Kz / entry.Kz
        T = np.array([
            [factor * np.abs(J_trans[0, 0]) ** 2, factor * np.abs(J_trans[0, 1]) ** 2],
            [factor * np.abs(J_trans[1, 0]) ** 2, factor * np.abs(J_trans[1, 1]) ** 2]
        ])
        return R, T


class Model(object):
    """
    This class and its children enable the user to construct ``Structures`` automatically from given parameters.
    The class ``Model`` can be viewed as an abstract class that defines parameters and methods common to all its
    children; however, it is possible to create an instance of ``Model``: it will have an empty ``Structure`` and its
    ``Layers`` (``Model.structure.layers``) can be added manually (``Structure.add_layer()``). The parameters of ``Models`` are:

    :param float n_entry: the refractive index of the stack’s entry isotropic semi-infinite medium
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param float wl_nm: the wavelength in nanometers
    :param float theta_in_rad: the angle of incidence in radians
    """

    # All common parameters initialised here
    # Children can call the parent's __init__ with super().__init__
    def __init__(self, n_entry, n_exit, wl_nm, theta_in_rad):
        self.n_entry = n_entry
        self.n_exit = n_exit
        self.wl = wl_nm
        self.theta_in = theta_in_rad
        theta_out = np.arcsin((n_entry / n_exit) * np.sin(self.theta_in + 0j))
        self.k0 = 2 * np.pi / self.wl
        self.Kx = self.n_entry * np.sin(self.theta_in)  # kx = Kx * k0
        self.Ky = 0  # ky = Ky * k0
        self.Kz_entry = self.n_entry * np.cos(self.theta_in)  # kz = Kz_entry * k0
        self.Kz_exit = self.n_exit * np.cos(theta_out)  # kz = Kz_entry * k0
        self.structure = self._build_structure_total()
        # When using super().__init__ in children classes, it's going to call the "_build_structure_total" method in the
        # parent class, which then will call both the "_build_structure" method of each child (as each child has a
        # "_build_structure" method that overrides the parent one), and "_build_entry_exit" of the parent (since it's not
        # overridden by a child version).

    def init_old(self, n_entry, n_exit, wl_nm, theta_in_rad):
        self.n_entry = n_entry
        self.n_exit = n_exit
        self.wl = wl_nm
        self.theta_in = theta_in_rad
        theta_out = np.arcsin((n_entry / n_exit) * np.sin(self.theta_in))  # theta_in + 0j
        self.k0 = 2 * np.pi / self.wl
        self.Kx = self.n_entry * np.sin(self.theta_in)  # kx = Kx * k0
        self.Ky = 0  # ky = Ky * k0
        self.Kz_entry = self.n_entry * np.cos(self.theta_in)  # kz = Kz_entry * k0
        self.Kz_exit = self.n_exit * np.cos(theta_out)  # kz = Kz_entry * k0
        self.structure = self._build_structure_total()
        # When using super().__init__ in children classes, it's going to call the "_build_structure_total" method in the
        # parent class, which then will call both the "_build_structure" method of each child (as each child has a
        # "_build_structure" method that overrides the parent one), and "_build_entry_exit" of the parent (since it's not
        # overridden by a child version).

    def copy_as_stack(self):
        """
        This function retrieves the permittivity and the thickness of the ``Structure`` created by the ``Model`` and
        creates an identical non-periodic ``StackModel`` (if the ``Model`` was periodic, the ``StackModel`` contains
        multiple times the same layers, but no periodic pattern to repeat).

        :return: a ``StackModel``
        """
        # TODO give the option to keep the periodicity or not
        # TODO make an unpack function to unpack the layers
        # TODO make a pack function where you give it a start and stop index
        N_per = self.structure.N_periods
        # Create an empty StackModel
        new_model = StackModel([], [], self.n_entry, self.n_exit, self.wl, self.theta_in, N_per=1)
        # Access the parameters of the layers of self
        eps_list = [lay.eps for lay in self.structure.layers]
        thickness_nm_list = [lay.thickness for lay in self.structure.layers]
        # Copy the parameters of the layers of self inside the new StackModel
        new_model.eps_list = eps_list.copy() * N_per
        new_model.thickness_nm_list = thickness_nm_list.copy() * N_per
        new_model.N_per = 1
        new_model.structure.layers = self.structure.layers.copy() * N_per
        return new_model

    def _build_structure_total(self):
        """
        This function creates the entry and exit ``HalfSpaces`` and the ``Structure`` of the ``Model``. The children
        classes of ``Model`` use this parent function, unless they redefine it. It calls their own (child)
        ``_build_structure()`` function, and ``Model``’s (parent) ``_build_entry_exit()``, unless the children redefine
        it.

        :return: a ``Structure``
        """
        entry_space, exit_space = self._build_entry_exit()
        structure = self._build_structure(entry_space, exit_space)
        return structure

    def _build_structure(self, entry_space, exit_space):
        """
        This function creates the ``Structure`` related to the ``Model``. The children classes of ``Model`` redefine
        their ``_build_structure()`` function.

        :return: a ``Structure``
        """
        warnings.warn("The build_function method of the Model class is used.")
        return Structure(entry=entry_space, exit=exit_space, Kx=self.Kx, Ky=self.Ky, Kz_entry=self.Kz_entry,
                         Kz_exit=self.Kz_exit, k0=self.k0,
                         N_periods=1)

    def _build_entry_exit(self):
        """
        This function creates the entry and exit ``HalfSpaces``. The children classes of ``Model`` use this parent
        function, unless they redefine it.

        :return: two ``HalfSpace`` objects for the isotropic entry and exit ``HalfSpaces``
        """
        epsilon_entry = np.array([[self.n_entry ** 2, 0, 0], [0, self.n_entry ** 2, 0], [0, 0, self.n_entry ** 2]])
        epsilon_exit = np.array([[self.n_exit ** 2, 0, 0], [0, self.n_exit ** 2, 0], [0, 0, self.n_exit ** 2]])
        entry_space = HalfSpace(epsilon_entry, self.Kx, self.Kz_entry, self.k0, category="isotropic")
        exit_space = HalfSpace(epsilon_exit, self.Kx, self.Kz_exit, self.k0, category="isotropic")
        return entry_space, exit_space

    def _build_entry_exit_old(self):
        """
        This function creates the entry and exit ``HalfSpaces``. The children classes of ``Model`` use this parent
        function, unless they redefine it.

        :return: two ``HalfSpace`` objects for the isotropic entry and exit ``HalfSpaces``
        """
        epsilon_entry = np.array([[self.n_entry ** 2, 0, 0], [0, self.n_entry ** 2, 0], [0, 0, self.n_entry ** 2]])
        epsilon_exit = np.array([[self.n_exit ** 2, 0, 0], [0, self.n_exit ** 2, 0], [0, 0, self.n_exit ** 2]])
        entry_space = HalfSpace(epsilon_entry, self.Kx, self.Kz_entry, self.k0, category="isotropic")
        exit_space = HalfSpace(epsilon_exit, self.Kx, self.Kz_exit, self.k0, category="isotropic")
        return entry_space, exit_space

    def get_refl_trans(self, circ=False, method="SM"):
        # TODO update docstring
        r"""
        This function calculates the ``Model``’s reflectance in the linear or circular polarisation basis, with the
        method chosen by the user.

        :param bool circ: ``False`` to express results in the linear polarisation basis, ``True`` to express results in
                          the circular polarisation basis
        :param string method: the matrix method to use for the calculation:

                              - ``"SM"`` for the scattering matrix method
                              - ``"TM"`` for the transfer matrix method with the eigenvectors and eigenvalues
                              - ``"EM"`` for the transfer matrix method with the direct exponential of Berreman’s matrix

        :return: reflectance: 2x2 Numpy array whose values correspond to:

                     - in the linear polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 R_{p \: \text{to} \: p} & R_{s \: \text{to} \: p} \\
                                 R_{p \: \text{to} \: s} & R_{s \: \text{to} \: s}
                             \end{bmatrix}

                    - in the circular polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 R_{RCP \: \text{to} \: RCP} & R_{LCP \: \text{to} \: RCP} \\
                                 R_{RCP \: \text{to} \: LCP} & R_{LCP \: \text{to} \: LCP}
                             \end{bmatrix}


        :return: transmittance: 2x2 Numpy array whose values correspond to:

                     - in the linear polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 T_{p \: \text{to} \: p} & T_{s \: \text{to} \: p} \\
                                 T_{p \: \text{to} \: s} & T_{s \: \text{to} \: s}
                             \end{bmatrix}

                    - in the circular polarisation basis (``circ=False``):

                         .. math::
                             \begin{bmatrix}
                                 T_{RCP \: \text{to} \: RCP} & T_{LCP \: \text{to} \: RCP} \\
                                 T_{RCP \: \text{to} \: LCP} & T_{LCP \: \text{to} \: LCP}
                             \end{bmatrix}

        """
        return self.structure.get_refl_trans(circ=circ, method=method)


class MixedModel(Model):
    """
    This class represents the combination of several ``Models`` with their sub-periodicities, given in a list of
    ``Models``. The entry and exit ``HalfSpaces`` of these ``Models`` are ignored and replaced
    by these of the ``MixedModel``. ``Kx`` and ``k0`` must be identical throughout all stacked models, which is checked
    at the initialisation; the ``Models`` that don't fit will be discarded and a warning will be issued.

    :param list models_list: a list of ``Models`` (``models_list[0]`` is on top of the stack, after the entry ``HalfSpace``)
    :param float n_entry: the refractive index of the stack’s entry isotropic semi-infinite medium
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param float wl_nm: the wavelength in nanometers
    :param float theta_in_rad: the angle of incidence in radians
    """

    def __init__(self, models_list, n_entry, n_exit, wl_nm, theta_in_rad):
        super().__init__(n_entry, n_exit, wl_nm, theta_in_rad)
        compatible_models = []
        kmod = 0
        for mod in models_list:
            if (mod.Kx == self.Kx) & (mod.k0 == self.k0):
                # Don't add models that are empty:
                if mod.structure.layers != [] and mod.structure.N_periods != 0:
                    compatible_models.append(mod)
                else:
                    msg = "The model number " + str(kmod) + " has not been added."
                    warnings.warn(msg)
            # Don't add models that don't have the same Kx and k0:
            else:
                msg = "The model number " + str(kmod) + " has not been added."
                warnings.warn(msg)
            kmod = kmod + 1
        self.models_list = compatible_models
        self.structures_list = [m.structure for m in self.models_list]

    def copy_as_stack(self):
        raise NotImplementedError(
            "The function copy_as_stack has not been implemented for the class MixedModel yet.")

    def get_refl_trans(self, circ=False, method="SM"):
        if method == "SM":
            reflectance, transmittance = Structure._get_refl_trans_multi_SM(self.structures_list, self.structure.entry,
                                                                            self.structure.exit, circ=circ)
        elif method == "TM":
            reflectance, transmittance = Structure._get_refl_trans_multi_TM(self.structures_list, self.structure.entry,
                                                                            self.structure.exit, circ=circ)
        elif method == "EM":
            reflectance, transmittance = Structure._get_refl_trans_multi_EM(self.structures_list, self.structure.entry,
                                                                            self.structure.exit, circ=circ)
        else:
            raise Exception('Invalid method (only SM, TM and EM allowed).')
        return reflectance, transmittance


class CholestericModel(Model):
    """
    This class represents a cholesteric liquid crystal with a multilayer stack of rotating nematic layers, constructed
    from a ``Cholesteric`` physical model.

    :param Cholesteric chole: a ``Cholesteric`` object from the ``Cholesteric`` library
    :param float n_e: the extraordinary refractive index
    :param float n_o: the ordinary refractive index
    :param float n_entry: the refractive index of the stack’s entry isotropic semi-infinite medium
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param int N_per: the number of periods. The cholesteric ``chole`` may already represent more than one helicoid: the layers created from the helicoid(s) in ``chole`` represent the periodic unit, which is repeated ``N_per`` times in ``CholestericModel``.
    :param float wl_nm: the wavelength in nanometers
    :param float theta_in_rad: the angle of incidence in radians
    """

    def __init__(self, chole, n_e, n_o, n_entry, n_exit, wl_nm, N_per, theta_in_rad):
        self.cholesteric = chole
        theta_in_rad_eff = theta_in_rad - chole.tilt
        self.n_e = n_e
        self.n_o = n_o
        self.N_per = N_per
        super().__init__(n_entry, n_exit, wl_nm, theta_in_rad_eff)

    def _build_structure(self, entry_space, exit_space):
        """
        This function is the routine that constructs the ``CholestericModel``’s ``Structure``.
        It overrides ``Model._build_structure()``.
        See ``Model.build_structure()`` for more information.
        """
        # Create an empty structure between isotropic half spaces
        chole_structure = Structure(entry_space, exit_space, self.Kx, self.Ky, self.Kz_entry, self.Kz_exit, self.k0,
                                    self.N_per)

        # Retrieve permittivity tensor for the first (not rotated) layer
        eps0 = np.array([[self.n_e ** 2, 0, 0],
                         [0, self.n_o ** 2, 0],
                         [0, 0, self.n_o ** 2]])

        # Create all layers of the 1-pitch structure with each its rotated permittivity
        pitch = []
        for ka in range(0, len(self.cholesteric.slicing)):
            # Retrieve the angular rotation and the thickness of the current layer
            angle_rad = self.cholesteric.slices_rotangles[ka]
            if ka == len(self.cholesteric.slicing) - 1:
                thickness = self.cholesteric.N_hel * self.cholesteric.pitch + self.cholesteric.slicing[0] - \
                            self.cholesteric.slicing[ka]
            else:
                thickness = self.cholesteric.slicing[ka + 1] - self.cholesteric.slicing[ka]
            eps = Layer.rotate_permittivity(eps0, angle_rad, axis='z')
            # Create the Layer
            layer = Layer(eps, thickness, self.Kx, self.k0)
            # Add the Layer to the pitch
            pitch.append(layer)
        # Store all the pitches in the Structure
        chole_structure.layers.extend(pitch)
        return chole_structure


class SlabModel(Model):
    """
    This class represents a homogeneous slab of arbitrary permittivity.

    :param ndarray eps: permittivity tensor, 3x3 Numpy array
    :param float thickness_nm: thickness in nanometers
    :param float n_entry: the refractive index of the stack’s entry isotropic semi-infinite medium
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param float wl_nm: the wavelength in nanometers
    :param float theta_in_rad: the angle of incidence in radians
    :param float rotangle_rad: the rotation angle to apply to the permittivity tensor, in radians
    :param ndarray rotaxis: the rotation axis, a one-dimensional Numpy array of length 3 (or the string ``'x'``, ``'y'`` or ``'z'``)
    """

    def __init__(self, eps, thickness_nm, n_entry, n_exit, wl_nm, theta_in_rad, rotangle_rad=0, rotaxis='z'):
        if rotangle_rad != 0:
            eps = Layer.rotate_permittivity(eps, rotangle_rad, rotaxis)
        self.eps = eps
        self.thickness = thickness_nm
        super().__init__(n_entry, n_exit, wl_nm, theta_in_rad)

    def _build_structure(self, entry_space, exit_space):
        """
        This function is the routine that constructs the ``SlabModel``’s ``Structure``.
        It overrides ``Model._build_structure()``.
        See ``Model.build_structure()`` for more information.
        """
        # Create an empty structure between isotropic half spaces
        slab_structure = Structure(entry_space, exit_space, self.Kx, self.Ky, self.Kz_entry, self.Kz_exit, self.k0)

        # Create the slab Layer
        slab_layer = Layer(self.eps, self.thickness, self.Kx, self.k0)
        slab_structure.layers.append(slab_layer)

        return slab_structure


class StackModel(Model):
    """
    This class represents a periodic multilayer stack where each layer has a given permittivity and thickness.

    :param list eps_list: list of permittivity tensors for each layer, each a 3x3 Numpy array
    :param list thickness_nm_list: list of thicknesses in nanometers for each layer, each a float
    :param float n_entry: the refractive index of the stack’s entry isotropic semi-infinite medium
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param float wl_nm: the wavelength in nanometers
    :param float theta_in_rad: the angle of incidence in radians
    :param int N_per: the number of periods
    """

    def __init__(self, eps_list, thickness_nm_list, n_entry, n_exit, wl_nm, theta_in_rad, N_per=1):
        if len(eps_list) != len(thickness_nm_list):
            raise Exception('The lists of permittivities and thicknesses must have the same length.')
        self.eps_list = eps_list
        self.thickness_list = thickness_nm_list
        self.N_per = N_per
        super().__init__(n_entry, n_exit, wl_nm, theta_in_rad)

    def _build_structure(self, entry_space, exit_space):
        """
        This function is the routine that constructs the ``StackModel``’s ``Structure``.
        It overrides ``Model._build_structure()``.
        See ``Model.build_structure()`` for more information.
        """
        # Create an empty structure between isotropic half spaces
        stack_structure = Structure(entry_space, exit_space, self.Kx, self.Ky, self.Kz_entry, self.Kz_exit, self.k0,
                                    self.N_per)

        # Create the stack Layers
        for k in range(len(self.eps_list)):
            layer = Layer(self.eps_list[k], self.thickness_list[k], self.Kx, self.k0)
            stack_structure.layers.append(layer)

        return stack_structure

    def change_N_per(self, new_N_per):
        """
        This function changes the number of periods of the Bragg stack.

        :param int new_N_per: new number of periods
        """
        self.N_per = new_N_per
        self.structure.N_periods = new_N_per

    def rotate_layer(self, layer_index, rot_angle_rad, rot_axis='z', hold=False):
        """
        This function rotates a given ``Layer``'s permittivity tensor: it creates the new rotated ``Layer`` and replaces
        the non-rotated ``Layer`` by the rotated ``Layer`` in the ``Structure``.

        :param int layer_index: the index of the ``Layer`` to rotate
        :param float rot_angle_rad: the rotation angle in radians
        :param ndarray rot_axis: the rotation axis
        :param bool hold: when the user decides to hold (``hold=True``) the calculation of Berreman’s matrix, the eigenvalues
        and eigenvectors, the user must then manually apply the functions to the ``Layer`` before calculating the
        transfer or scattering matrix. This is exceptional practice. The default is ``hold=True``.
        """
        # Make a new layer with rotated permittivity
        epsilon = self.structure.layers[layer_index].eps
        thickness_nm = self.structure.layers[layer_index].thickness
        Kx = self.structure.layers[layer_index].Kx
        k0 = self.structure.layers[layer_index].k0
        new_layer = Layer(epsilon, thickness_nm, Kx, k0, rot_angle_rad=rot_angle_rad, rot_axis=rot_axis, hold=hold)
        # Put the new layer in the stack
        self.structure.replace_layer(layer_index, new_layer)
        # Update the permittivity list
        self.eps_list[layer_index] = self.structure.layers[layer_index].eps

    def rotate_layers(self, layer_number_list, rot_angle_rad_list, rot_axis='z'):
        """
        This function applies the function ``rotate_layer`` on several ``Layers``. See ``rotate_layer``’s documentation.
        """
        if layer_number_list == 'all':
            layer_number_list = list(np.arange(len(self.structure.layers)))
        if isinstance(rot_angle_rad_list, int):
            rot_angle_rad_list = [rot_angle_rad_list] * len(layer_number_list)
        if isinstance(rot_angle_rad_list, float):
            rot_angle_rad_list = [rot_angle_rad_list] * len(layer_number_list)
        for k in range(len(layer_number_list)):
            self.rotate_layer(layer_number_list[k], rot_angle_rad_list[k], rot_axis=rot_axis)

    def add_layer(self, new_layer):
        """
        This function adds a ``Layer`` to the multilayer stack represented by the ``StackModel``.

        :param Layer new_layer: ``Layer`` to add
        """
        self.structure.add_layer(new_layer)
        self.eps_list.append(new_layer.eps)
        self.thickness_list.append(new_layer.thickness)

    def add_layers(self, new_layers_list):
        """
        This function adds a list of ``Layers`` to the multilayer stack represented by the ``StackModel``.

        :param list new_layers_list: list of ``Layers`` to add
        """
        compatible_layers = self.structure.add_layers(new_layers_list)
        new_eps_list = [lay.eps for lay in compatible_layers]
        new_thick_list = [lay.thickness for lay in compatible_layers]
        self.eps_list.extend(new_eps_list)
        self.thickness_list.append(new_thick_list)

    def extract_stack(self, index_first_layer, index_last_layer):
        """
        This function extracts a sub-stack from the ``StackModel`` (and return a new instance of ``StackModel``).

        :param index_first_layer: index of the first ``Layer``
        :param index_last_layer: index of the last ``Layer`` to extract + 1 (if ``index_first_layer = index_last_layer``, the
        sub-stack contains the ``Layer`` indexed ``index_first_layer``)
        :return: a ``StackModel``
        """
        extracted_stack = StackModel([], [], self.n_entry, self.n_exit, self.wl, self.theta_in, self.N_per)
        layers_to_extract = self.structure.layers[index_first_layer:index_last_layer]
        extracted_stack.add_layers(layers_to_extract)
        return extracted_stack


class StackOpticalThicknessModel(Model):
    """
    This class represents a periodic multilayer stack where all layers are isotropic and have the same optical thickness.

    :param list n_list: list of refractive indices for each ``Layer``, each a float
    :param float total_thickness_nm: totat thickness of the stack, in nanometers
    :param float n_entry: the refractive index of the stack’s entry isotropic semi-infinite medium
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param float wl_nm: the wavelength in nanometers
    :param float theta_in_rad: the angle of incidence in radians
    :param int N_per: the number of periods
    """

    def __init__(self, n_list, total_thickness_nm, n_entry, n_exit, wl_nm, theta_in_rad, N_per=1):
        self.n_list = n_list
        self.L = total_thickness_nm
        self.N_per = N_per
        self.phi = np.NaN  # optical thickness
        super().__init__(n_entry, n_exit, wl_nm, theta_in_rad)

    def _build_structure(self, entry_space, exit_space):
        """
        This function is the routine that constructs the ``StackOpticalThicknessModel``’s ``Structure``.
        It overrides ``Model._build_structure()``.
        See ``Model.build_structure()`` for more information.
        """
        # Create an empty structure between isotropic half spaces
        stack_structure = Structure(entry_space, exit_space, self.Kx, self.Ky, self.Kz_entry, self.Kz_exit, self.k0,
                                    self.N_per)

        eps_list = [np.array([[ni ** 2, 0, 0], [0, ni ** 2, 0], [0, 0, ni ** 2]]) for ni in self.n_list]

        phi = 1
        thickness_list_temp = [self.wl * phi / (2 * np.pi * ni) for ni in self.n_list]
        thickness_list_optical = [t * self.L / sum(thickness_list_temp) for t in thickness_list_temp]
        thickness_list_optical.append(self.L / sum(thickness_list_optical))
        thickness_list = thickness_list_optical
        self.phi = 2 * np.pi * self.n_list[0] * thickness_list_optical[0] / self.wl

        # Create the stack Layers
        for k in range(len(eps_list)):
            slab_layer = Layer(eps_list[k], thickness_list[k], self.Kx, self.k0)
            stack_structure.layers.append(slab_layer)

        return stack_structure


class Spectrum(object):
    """
    This class implements the modelling of a multilayer stack over a range of wavelength and provide tools for calculating
    reflection spectra with the choice of the polarisation basis and for exporting the data. ``Spectrum`` contains an
    initially-empty dictionary ``Spectrum.data`` that will be filled with the calculated reflection spectra and additional data.

    :param ndarray wl_nm_list: a list (or array or range) of wavelengths (integers or floats), for example ``range(400, 800)``
    :param string model_type: the name of the model to use. These include:

        - ``"CholestericModel"``
        - ``"SlabModel"``
        - ``"StackModel"``
        - ``"StackOpticalThicknessModel"``

    :param dict model_parameters: a dictionary containing the list of parameters required to create the chosen ``Model``,
    except the wavelength. See the chosen ``Model``’s documentation to know how to construct the dictionary.
    """

    def __init__(self, wl_nm_list, model_type, model_parameters):
        self.wl_list = wl_nm_list  # list of wavelengths in nm
        self.mo_type = model_type  # name of model (cholesteric, slab, stack...)
        self.mo_param = model_parameters  # dictionary with model parameters
        self.model = []
        self.data = {}  # empty dictionary; all calculated data will go here

    def export(self, path_out, with_param=True):
        """
        This function exports the ``Spectrum`` for further processing in MATLAB or Python, and stores it to the specified path.
        The contents of ``Spectrum.data`` and ``Spectrum.wl_list`` are exported.

        :param string path_out: path of the file to save the spectrum. It must end with ``".mat"`` (to save in
         MATLAB-compatible format) or ``".pck"`` (to save with Pickles in Python-compatible format).
        :param bool with_param: ``True`` to save the ``Spectrum``’s model parameters in addition to the content of ``Spectrum.data``,
        ``False`` (default) to only save the content of ``Spectrum.data``.
        """
        if with_param:
            dict_to_save = {**self.mo_param, **self.data,
                            "wl": self.wl_list}  # merge the dictionaries with data to export (Python 3.5 and above)
        else:
            dict_to_save = {**self.data, "wl": self.wl_list}
        # dict_to_save['wl'] = self.wl_list
        if len(path_out) > 4:
            if path_out[-4:] == ".pck":
                with open(path_out, 'wb') as f:
                    pickle.dump(dict_to_save, f)
                f.close()
            elif path_out[-4:] == ".mat":
                scipy.io.savemat(path_out, dict_to_save)
            else:
                raise Exception('Invalid file format, must end with .mat or .pck.')
        else:
            raise Exception('Invalid file name, must contain at least one character.')

    @staticmethod
    def create_model(model_type, model_parameters, wl):
        # Create the appropriate Model for each wavelength
        if model_type == "CholestericModel":
            new_model = CholestericModel(model_parameters['chole'],
                                         model_parameters['n_e'],
                                         model_parameters['n_o'],
                                         model_parameters['n_entry'],
                                         model_parameters['n_exit'], wl,
                                         model_parameters['N_per'],
                                         model_parameters['theta_in_rad'])
        elif model_type == "SlabModel":
            default_param = dict(rotangle_rad=0, rotaxis='z')
            model_parameters = {**default_param,
                                **model_parameters}  # model_parameters is added to default_param and will overwrite
            new_model = SlabModel(model_parameters['eps'],
                                  model_parameters['thickness_nm'],
                                  model_parameters['n_entry'],
                                  model_parameters['n_exit'], wl,
                                  model_parameters['theta_in_rad'],
                                  model_parameters['rotangle_rad'],
                                  model_parameters['rotaxis'])
        elif model_type == "StackModel":
            default_param = dict(N_per=1)
            model_parameters = {**default_param,
                                **model_parameters}  # model_parameters is added to default_param and will overwrite
            new_model = StackModel(model_parameters['eps_list'],
                                   model_parameters['thickness_nm_list'],
                                   model_parameters['n_entry'],
                                   model_parameters['n_exit'],
                                   wl,
                                   model_parameters['theta_in_rad'],
                                   model_parameters['N_per'])
        elif model_type == "StackOpticalThicknessModel":
            default_param = dict(N_per=1)
            model_parameters = {**default_param,
                                **model_parameters}  # model_parameters is added to default_param and will overwrite
            new_model = StackOpticalThicknessModel(model_parameters['n_list'],
                                                   model_parameters['total_thickness_nm'],
                                                   model_parameters['n_entry'],
                                                   model_parameters['n_exit'],
                                                   wl,
                                                   model_parameters['theta_in_rad'],
                                                   model_parameters['N_per'])
        # Add your new Model here
        # elif self.motype == "YourNewModel":
        #    raise NotImplementedError("The Spectrum calculation has not been implemented for YourNewModel yet.")
        else:
            raise Exception('Invalid model type.')
        return new_model

    def calculate_refl_trans(self, circ=False, method="SM", talk=False):
        """
        This function creates the required ``Model`` and calculates the reflection spectrum in the linear (default) or
        circular polarisation basis, usinq a chosen method (by default the scattering matrix method). The results are
        stored in the initially empty dictionary ``Spectrum.data``. The values of the results correspond to:

        - in the linear polarisation basis (``circ=False``):

            - ``Spectrum.data["R_p_to_p"]``: reflection spectrum for incoming :math:`p`-polarisation to outgoing :math:`p`-polarisation, 1d Numpy array
            - ``Spectrum.data["R_p_to_s"]``: reflection spectrum for incoming :math:`p`-polarisation to outgoing :math:`s`-polarisation, 1d Numpy array
            - ``Spectrum.data["R_s_to_p"]``: reflection spectrum for incoming :math:`s`-polarisation to outgoing :math:`p`-polarisation, 1d Numpy array
            - ``Spectrum.data["R_s_to_s"]``: reflection spectrum for incoming :math:`s`-polarisation to outgoing :math:`s`-polarisation, 1d Numpy array

        - in the circular polarisation basis (``circ=False``):

            - ``Spectrum.data["R_R_to_R"]``: reflection spectrum for incoming :math:`RCP`-polarisation to outgoing :math:`RCP`-polarisation, 1d Numpy array
            - ``Spectrum.data["R_R_to_L"]``: reflection spectrum for incoming :math:`RCP`-polarisation to outgoing :math:`LCP`-polarisation, 1d Numpy array
            - ``Spectrum.data["R_L_to_R"]``: reflection spectrum for incoming :math:`LCP`-polarisation to outgoing :math:`RCP`-polarisation, 1d Numpy array
            - ``Spectrum.data["R_L_to_L"]``: reflection spectrum for incoming :math:`LCP`-polarisation to outgoing :math:`LCP`-polarisation, 1d Numpy array

        as well as ``time_elapsed`` (float) which calculates the time that it took to compute the spectrum.

        :param bool circ: ``False`` to express results in the linear polarisation basis, ``True`` to express results in
                          the circular polarisation basis
        :param string method: the matrix method to use for the calculation:

                              - ``"SM"`` for the scattering matrix method
                              - ``"TM"`` for the transfer matrix method with the eigenvectors and eigenvalues
                              - ``"EM"`` for the transfer matrix method with the direct exponential of Berreman’s matrix
        :param bool talk: ``True`` (non-default) to display the computation progress, wavelength per wavelength
        """
        t = time.time()
        # Create empty variables
        r_00 = []
        r_01 = []
        r_10 = []
        r_11 = []
        t_00 = []
        t_01 = []
        t_10 = []
        t_11 = []
        # Browse through all wavelengths
        for wl in self.wl_list:
            # If mo_type and mo_param is not a list, the Spectrum is for one Model only
            if not isinstance(self.mo_type, list):
                model = Spectrum.create_model(self.mo_type, self.mo_param, wl)
            # If mo_type and mo_param are lists, the Spectrum is for a MixedModel
            else:
                partial_models_list = []
                for km in range(len(self.mo_type)):
                    partial_model = Spectrum.create_model(self.mo_type[km], self.mo_param[km], wl)
                    partial_models_list.append(partial_model)
                model = MixedModel(partial_models_list,
                                   self.mo_param[0]['n_entry'],
                                   self.mo_param[0]['n_exit'],
                                   wl,
                                   self.mo_param[0]['theta_in_rad'])
                # The MixedModel class will check that the angles of incidence, n_entry and n_exit are the same for
                # all models in the list, therefore it is not necessary to check again here.
            self.model.append(model)

            # Calculate the reflectance using the chosen method and for the given polarisation basis
            reflectance, transmittance = model.get_refl_trans(method=method, circ=circ)

            # Add the reflectance to lists
            r_00.append(reflectance[0, 0])
            r_01.append(reflectance[0, 1])
            r_10.append(reflectance[1, 0])
            r_11.append(reflectance[1, 1])
            t_00.append(transmittance[0, 0])
            t_01.append(transmittance[0, 1])
            t_10.append(transmittance[1, 0])
            t_11.append(transmittance[1, 1])

            if talk:
                print("Wavelength " + str(wl) + " nm done.")

        elapsed = time.time() - t

        # Save the lists in the data dictionary
        # If there's already a variable with the same key, it'll overwrite it
        if circ:
            self.data['R_R_to_R'] = np.array(r_00)
            self.data['R_L_to_R'] = np.array(r_01)
            self.data['R_R_to_L'] = np.array(r_10)
            self.data['R_L_to_L'] = np.array(r_11)
            self.data['T_R_to_R'] = np.array(t_00)
            self.data['T_L_to_R'] = np.array(t_01)
            self.data['T_R_to_L'] = np.array(t_10)
            self.data['T_L_to_L'] = np.array(t_11)
        else:
            self.data['R_p_to_p'] = np.array(r_00)
            self.data['R_s_to_p'] = np.array(r_01)
            self.data['R_p_to_s'] = np.array(r_10)
            self.data['R_s_to_s'] = np.array(r_11)
            self.data['T_p_to_p'] = np.array(t_00)
            self.data['T_s_to_p'] = np.array(t_01)
            self.data['T_p_to_s'] = np.array(t_10)
            self.data['T_s_to_s'] = np.array(t_11)

        self.data['time_elapsed'] = elapsed

    def rename_result(self, old_key, new_key):
        """
        This function enables to rename one of the elements contained in ``Spectrum.data``.

        For example, the user may calculate the reflection spectrum with the scattering matrix method:
        ::
            my_spectrum.calculate(method="SM")

        then rename the keys in ``Spectrum.data`` with:
        ::
            my_spectrum.rename_result("R_R_to_R", "R_R_to_R_SM")
            my_spectrum.rename_result("R_R_to_L", "R_R_to_L_SM")
            my_spectrum.rename_result("R_L_to_R", "R_L_to_R_SM")
            my_spectrum.rename_result("R_L_to_L", "R_L_to_L_SM")
            my_spectrum.rename_result("time_elapsed", "time_elpased_SM")

        then calculate the reflection spectrum with the transfer matrix method:
        ::
            Spectrum.calculate(method="TM")

        then rename the keys in ``Spectrum.data`` with:
        ::
            my_spectrum.rename_result("R_R_to_R", "R_R_to_R_TM")
            my_spectrum.rename_result("R_R_to_L", "R_R_to_L_TM")
            my_spectrum.rename_result("R_L_to_R", "R_L_to_R_TM")
            my_spectrum.rename_result("R_L_to_L", "R_L_to_L_TM")
            my_spectrum.rename_result("time_elapsed", "time_elpased_TM")

        in order to save results calculated with both matrix methods.
        """
        self.data[new_key] = self.data.pop(old_key)

    def add_result(self, key, value):
        """
        This function add an entry to the dictionary ``Spectrum.data``. This entry will be included in the content
        that is saved by ``Spectrum.export()``.

        :param string key: the key for the value to add to the dictionary
        :param value: the value to add (any type)
        """
        self.data[key] = value
