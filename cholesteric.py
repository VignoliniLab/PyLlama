import numpy as np
from numpy import cos, sin, tan, arctan2, sqrt
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # important for 3D plots
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import proj3d
from scipy.linalg import norm


# vec_u = unit vector
x_u = np.array([1, 0, 0])
y_u = np.array([0, 1, 0])
z_u = np.array([0, 0, 1])


def set_axes_equal(ax, manual_lims=None):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc.
    From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    :param ax: a Matplotlib axis, e.g., as output from plt.gca().
    """

    if manual_lims == None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    else:
        x_limits = manual_lims[0]
        y_limits = manual_lims[1]
        z_limits = manual_lims[2]

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class Cholesteric(object):
    """
    This class represents the directors of a cholesteric liquid crystal with a multilayer stack of rotating nematic
    layers. The physical model can be found in Frka-Petesic et al, Physical Review Materials, 2019, doi:10.1103/PhysRevMaterials.3.045601

    :param int pitch360: the (full) pitch for a 360 degree rotation in nanometers
    :param float tilt_rad: the tilt of the helical axis in radians
    :param int resolution: the number of planes over a 360 degree rotation
    :param int handedness: the value of ``handedness`` is ``+1`` for right-handed and ``-1`` for left-handed
    :param float n_exit: the refractive index of the stack’s exit isotropic semi-infinite medium
    :param int N_hel360: the number of full pitches to stack
    """

    def __init__(self, pitch360=500, tilt_rad=0, resolution=40, handedness=1, N_hel360=1):
        self.resolution = resolution
        self.handedness = handedness
        self.N_hel = N_hel360
        self.pitch = pitch360  # length in nm for a 360deg rotation
        self.q = handedness * 2 * np.pi / self.pitch
        self.tilt = tilt_rad  # tilt, angle in degrees between 0 and 90
        self.slicing = np.linspace(0, self.N_hel * pitch360, self.N_hel * self.resolution, endpoint=False)
        self.e1_u = np.array([cos(self.beta), 0, -sin(self.beta)])
        self.e2_u = np.array([0, 1, 0])
        self.e3_u = np.array([sin(self.beta), 0, cos(self.beta)])
        self.helical_axis = np.array([sin(self.tilt), 0, cos(self.tilt)])  # helical axis
        self.slices_rotangles = self.q * self.slicing
        self.slices_directors = [cos(self.tilt)*cos(iphi)*x_u + sin(iphi)*y_u -sin(self.tilt)*cos(iphi)*z_u for iphi in self.slices_rotangles]
        self.compression = 1
        self.history = []

    def copy(self):
        ch = Cholesteric()
        ch.resolution = self.resolution
        ch.pitch = self.pitch
        ch.q = self.q
        ch.tilt = self.tilt
        ch.slicing = self.slicing
        #ch.e1_u = self.e1_u
        #ch.e2_u = self.e2_u
        #ch.e3_u = self.e3_u
        ch.helical_axis = self.helical_axis
        ch.slices_rotangles = self.slices_rotangles
        ch.slices_directors = self.slices_directors
        return ch

    def compress(self, alpha):
        """
        This function updates the fields of the ``Cholesteric`` after vertical compression. The physical model can be
        found in Frka-Petesic et al, Physical Review Materials, 2019, doi:10.1103/PhysRevMaterials.3.045601

        :param float alpha: the coefficient of vertical compression, between 0 and 1
        """
        e1_u_new = np.array([cos(self.tilt), 0, -alpha*sin(self.tilt)])
        e2_u_new = np.array([0, 1, 0])
        e3_u_new = np.array([sin(self.tilt), 0, alpha*cos(self.tilt)])
        m_u_new = np.array([alpha*sin(self.tilt), 0, cos(self.tilt)])  # beta before compression
        beta_new = arctan2(alpha*sin(self.tilt), cos(self.tilt))
        p_new = self.pitch * sqrt(sin(beta_new)**2 + alpha**2*cos(beta_new)**2)
        phi_new = arctan2(p_new*sin(self.slices_rotangles), cos(self.slices_rotangles)*alpha*self.pitch)
        n_u_new = [cos(beta_new)*cos(iphi_new)*x_u + sin(iphi_new)*y_u + -sin(beta_new)*cos(iphi_new)*z_u for iphi_new in phi_new]
        # Save everything
        self.pitch = p_new
        self.q = 2 * np.pi / p_new
        self.tilt = beta_new
        self.slicing = np.linspace(0, self.N_hel * p_new, self.N_hel * self.resolution, endpoint=False)
        self.helical_axis = m_u_new
        self.slices_rotangles = phi_new
        self.slices_directors = n_u_new
        self.history.append('self.compress(' + str(alpha) + ')')

    def change_pitch(self, p_new):
        self.slicing = np.linspace(0, self.N_hel * p_new, self.N_hel * self.resolution, endpoint=False)
        self.pitch = p_new
        self.history.append('self.change_pitch(' + str(p_new) + ')')

    def change_tilt(self, beta_new):
        m_u_new = np.array([sin(beta_new), 0, cos(beta_new)])
        n_u_new = [cos(beta_new) * cos(iphi) * x_u + sin(iphi) * y_u + -sin(beta_new) * cos(iphi) * z_u for
                   iphi in self.slices_rotangles]
        self.helical_axis = m_u_new
        self.slices_directors = n_u_new
        self.tilt = beta_new
        self.history.append('self.change_tilt(' + str(beta_new) + ')')

    def distort(self, alpha):
        new_rotangles = []
        for phi in self.slices_rotangles:
            new_phi = np.arctan(alpha * np.tan(phi))
            new_rotangles.append(new_phi)
        self.slices_rotangles = new_rotangles
        self.slices_directors = [
                cos(self.tilt) * cos(iphi) * x_u + sin(iphi) * y_u - sin(self.tilt) * cos(iphi) * z_u for iphi in
                self.slices_rotangles]
        self.history.append('self.distort(' + str(alpha) + ')')

    def plot_phi(self):
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        plt.plot(self.slicing, self.slices_rotangles)
        ax.set_xlabel('z-axis (nm)')
        ax.set_ylabel('phi')
        return fig, ax

    def plot_xproj(self):
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        plt.plot(self.slicing, np.cos(self.slices_rotangles))
        ax.set_xlabel('z-axis (nm)')
        ax.set_ylabel('x-axis')
        return fig, ax

    def plot_yproj(self):
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        plt.plot(self.slicing, np.sin(self.slices_rotangles))
        ax.set_xlabel('z-axis (nm)')
        ax.set_ylabel('y-axis')
        return fig, ax

    def plot_xyproj(self):
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(np.cos(self.slices_rotangles), np.zeros(self.slicing.shape), self.slicing)
        plt.plot(np.zeros(self.slicing.shape), np.sin(self.slices_rotangles), self.slicing)
        return fig, ax

    def plot_simple_3D(self, fig=None, ax=None, view='classic'):
        import geometry
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        #ax.set_aspect('equal')
        for k in np.arange(0, len(self.slicing), 1):
            sl = self.slicing[k]
            n_u = self.slices_directors[k]
            ctr = sl * self.helical_axis
            cyl = geometry.Cylinder(centre=ctr, direction=n_u, length=150,
                                    radius=12, color='white')
            # radius = 6 is usually good, radius = 12 for the small paper fig with resolution = 12
            # length = 200 is usually good
            cyl.plot(ax, resolution=10)
        set_axes_equal(ax)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        if view == "classic":
            ax.view_init(180 + 20, 180 + 70)
        elif view == "front":
            ax.view_init(180, 90)  # front view
        elif view == "top":
            ax.view_init(270, 0)  # top view
        else:
            raise Exception("Invalid view.")
        # Set the spines in a nicer way:
        # https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot
        ax.xaxis._axinfo['juggled'] = (2, 0, 1)
        ax.yaxis._axinfo['juggled'] = (2, 1, 0)
        set_axes_equal(ax)
        return fig, ax

    def plot_simple(self, fig=None, ax=None, view='classic', type='3D'):
        import geometry
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        #ax.set_aspect('equal')
        if type == "2D":
            for k in np.arange(0, len(self.slicing), 1):
                sl = self.slicing[k]  # slice of the current rod
                n_u = self.slices_directors[k]  # direction of the current rod
                ctr = sl * self.helical_axis  # depth of the current rod
                x, y, z = geometry.line_middle_direction(ctr, n_u, 100)
                ax.plot3D(x, y, z, linewidth=1, alpha=0.7)
        elif type == "3D":
            for k in np.arange(0, len(self.slicing), 1):
                sl = self.slicing[k]
                n_u = self.slices_directors[k]
                ctr = sl * self.helical_axis
                cyl = geometry.Cylinder(centre=ctr, direction=n_u, length=200,
                                        radius=12, color='white')
                # radius = 6 is usually good, radius = 12 for the small paper fig with resolution = 12
                # length = 200 is usually good
                # or length = 150
                cyl.plot(ax, resolution=10)
        elif type == "arrow":
            x, y, z = geometry.line_point_direction([0, 0, 0], self.helical_axis, self.pitch * self.N_hel)
            plt.plot(x, y, z, 'r-', alpha=0)
            for k in np.arange(0, len(self.slicing), 1):
                sl = self.slicing[k]  # slice of the current rod
                n_u = self.slices_directors[k]  # direction of the current rod
                ctr = sl * self.helical_axis  # depth of the current rod
                x, y, z = geometry.line_middle_direction(ctr, n_u, 160)
                a = geometry.Arrow3D([x[1], x[0]], [y[1], y[0]], [z[1], z[0]], arrowstyle="-|>", mutation_scale=10,
                                     lw=1, color='k')
                ax.add_artist(a)
        else:
            raise Exception("Invalid type. Only '2D', '3D' and 'arrow' allowed.")
        set_axes_equal(ax)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        if view == "classic":
            ax.view_init(180 + 20, 180 + 70)
        elif view == "front":
            ax.view_init(180, 90)  # front view
        elif view == "top":
            ax.view_init(270, 0)  # top view
        else:
            raise Exception("Invalid view.")
        # Set the spines in a nicer way:
        # https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot
        ax.xaxis._axinfo['juggled'] = (2, 0, 1)
        ax.yaxis._axinfo['juggled'] = (2, 1, 0)
        return fig, ax

    def plot_simple_arrows(self, fig=None, ax=None, view='classic'):
        import geometry
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        #ax.set_aspect('equal')
        x, y, z = geometry.line_point_direction([0, 0, 0], self.helical_axis, self.pitch * self.N_hel)
        plt.plot(x, y, z, 'r-', alpha=0)
        for k in np.arange(0, len(self.slicing), 1):
            sl = self.slicing[k]  # slice of the current rod
            n_u = self.slices_directors[k]  # direction of the current rod
            ctr = sl * self.helical_axis  # depth of the current rod
            x, y, z = geometry.line_middle_direction(ctr, n_u, 160)
            a = geometry.Arrow3D([x[1], x[0]], [y[1], y[0]], [z[1], z[0]], arrowstyle="-|>", mutation_scale=10, lw=1, color='k')
            ax.add_artist(a)
        set_axes_equal(ax)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        if view == "classic":
            ax.view_init(180 + 20, 180 + 70)
        elif view == "front":
            ax.view_init(180, 90)  # front view
        elif view == "top":
            ax.view_init(270, 0)  # top view
        else:
            raise Exception("Invalid view.")
        # Set the spines in a nicer way:
        # https://stackoverflow.com/questions/48442713/move-spines-in-matplotlib-3d-plot
        ax.xaxis._axinfo['juggled'] = (2, 0, 1)
        ax.yaxis._axinfo['juggled'] = (2, 1, 0)
        return fig, ax

    def plot_add_optics(self, fig, ax, theta_in_rad):
        import geometry
        # Calculate the angles to display
        theta_in_eff_deg = round((theta_in_rad - self.tilt) * 180 / np.pi, 2)
        theta_in_deg = round(theta_in_rad * 180 / np.pi, 2)
        tilt_deg = round(self.tilt * 180 / np.pi, 2)
        # Origin
        por = np.array([0, 0, 0])
        # z axis in black
        x, y, z = geometry.line_point_direction(por, np.array([0, 0, 1]), 1.5 * self.pitch)
        ax.plot3D(x, y, z, linewidth=1, color='k', linestyle='dashed', label='z axis')  # color='k'
        x, y, z = geometry.line_point_direction(por, np.array([0, 0, 1]), -self.pitch)
        ax.plot3D(x, y, z, linewidth=1, color='k', linestyle='dashed')  # color='k'
        # Plot the director of the helicoid in blue
        x, y, z = geometry.line_point_direction(por, self.helical_axis, 1.5 * self.pitch)
        ax.plot3D(x, y, z, linewidth=1, color='b', linestyle='dashed', label='helix director')
        x, y, z = geometry.line_point_direction(por, self.helical_axis, -self.pitch)
        ax.plot3D(x, y, z, linewidth=1, color='b', linestyle='dashed')
        # Visualise the angle of the director in blue
        theta_mesh = np.linspace(0, self.tilt, 100)
        y = np.linspace(0, 0, 100)
        r = 50
        x = -r * np.sin(theta_mesh)
        z = -r * np.cos(theta_mesh)
        ax.plot(x, y, z, color='b', linewidth=1, label=(r'$\beta = $' + str(tilt_deg) + '°'))
        # Plot the direction of the angle of incidence in red
        dir = np.array([sin(theta_in_rad), 0, cos(theta_in_rad)])
        x, y, z = geometry.line_point_direction(por, -dir, self.pitch)
        ax.plot3D(x, y, z, linewidth=1, color='r', linestyle='dashed', label='incident light')  # color='r'
        # Visualise the angle of incidence in red
        theta_mesh = np.linspace(0, theta_in_rad, 100)
        y = np.linspace(0, 0, 100)
        r = 100
        x = -r * np.sin(theta_mesh)
        z = -r * np.cos(theta_mesh)
        ax.plot(x, y, z, color='r', linewidth=1, label=(r'$\theta_{in} = $' + str(theta_in_deg) + '°'))  # color='r'
        # Visualise the effective angle with respect to the helicoid in green
        theta_mesh = np.linspace(theta_in_rad, self.tilt, 100)
        y = np.linspace(0, 0, 100)
        r = 200  # 200 is default
        x = -r * np.sin(theta_mesh)
        z = -r * np.cos(theta_mesh)
        ax.plot(x, y, z, color='g', linewidth=2, label=(r'$\theta_{in\:eff} = $' + str(theta_in_eff_deg) + '°'))
        # Visualise the effective angle of incidence with a surface in green
        r_mesh = np.linspace(0, 200, 100)  # 200 is default
        theta_mesh = np.linspace(theta_in_rad, self.tilt, 100)
        x = np.outer(r_mesh, np.sin(theta_mesh))
        z = np.outer(r_mesh, np.cos(theta_mesh))
        s = x.shape
        y = np.zeros(s)
        ax.plot_surface(-x, y, -z, color='g')
        # Specify legend position (tune the numbers manually)
        # https://stackoverflow.com/questions/48442786/legend-specifying-3d-position-in-3d-axes-matplotlib
        set_axes_equal(ax)
        f = lambda x, y, z: proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
        #ax.legend(loc="lower left", bbox_to_anchor=f(-300, 1000, 300), bbox_transform=ax.transData)
        #ax.legend()
        set_axes_equal(ax)
        return fig, ax

    @staticmethod
    def fun(x, y):
        return x ** 2 + y


