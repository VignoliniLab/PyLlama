"""
Geometry
====

Provides
  A visual representation of cylinders in a
  Matplotlib 3D plot.


Class
  Cylinder: contains the parameters of the
  cylinder (centre, axis, length...) and
  enables the manipulation of cylinders
  and their representation in 3D volumes
  with user-chosen resolution.

Author
  Mélanie Bay (mmb54@cam.ac.uk)
"""


import random as rnd
import numpy as np
from scipy.linalg import norm
from warnings import warn
from matplotlib.patches import FancyArrowPatch   # for the arrows
from mpl_toolkits.mplot3d import proj3d


def set_axes_equal(ax, manual_lims=None):
    """
    From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    :param ax: a Matplotlib axis, e.g., as output from plt.gca().
    :return:
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


def line_two_points(p1, p2):
    """
    Returns the x, y, z to use in plot3D(x, y, z) to plot a line between two points
    :param p1: Numpy array of the first point, coordinates x, y, z
    :param p2: Numpy array of the second point, coordinates x, y, z
    :return: x, y, z to plot
    """
    return [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]


def line_point_direction(p1, dir, length=1):
    """
    Returns the x, y, z to use in plot3D(x, y, z) to plot a line given a starting point, a direction and a length
    """
    p2 = p1 + length * dir
    return line_two_points(p1, p2)


def line_middle_direction(p1, dir, length=1):
    """
    Returns the x, y, z to use in plot3D(x, y, z) to plot a line given a point in the middle of the line, a direction and a length
    """
    p2 = p1 + length * dir
    p0 = p1 - length * dir
    return line_two_points(p0, p2)


def translate_point(p1, dir, length=1):
    """
    Returns a translated point given a starting point, a direction and a length
    """
    p2 = p1 + length * dir
    return p2


class Cylinder(object):

    def __init__(self, centre=np.array([0, 0, 0]), direction=np.array([0, 0, 1]), length=1, radius=0.25, color='white'):
        self.centre = centre
        self.direction = direction / norm(direction)
        self.length = length
        self.radius = radius
        self.color = color
        self.ptsTubX = None  # Meshed points
        self.ptsTubY = None  # Meshed points
        self.ptsTubZ = None  # Meshed points
        self.ptsTopX = None  # Meshed points
        self.ptsTopY = None  # Meshed points
        self.ptsTopZ = None  # Meshed points
        self.ptsBotX = None  # Meshed points
        self.ptsBotY = None  # Meshed points
        self.ptsBotZ = None  # Meshed points

    @property
    def centre(self):
        return self.__centre

    @centre.setter
    def centre(self, centre):
        self.__centre = centre
        self.delPoints()  # The points don't match the structure anymore: delete!

    @property
    def direction(self):
        return self.__direction

    @direction.setter
    def direction(self, direction):
        self.__direction = direction / norm(direction)
        self.delPoints()  # The points don't match the structure anymore: delete!

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, length):
        if length < 0:
            length = - length
            warn("Negative Cylinder length. Absolute value has been taken.")
        self.__length = length
        self.delPoints()  # The points don't match the structure anymore: delete!

    @property
    def radius(self):
        return self.__radius

    @radius.setter
    def radius(self, radius):
        if radius < 0:
            radius = - radius
            warn("Negative Cylinder radius. Absolute value has been taken.")
        self.__radius = radius
        self.delPoints()  # The points don't match the structure anymore: delete!

    def delPoints(self):
        self.ptsTubX = None
        self.ptsTubY = None
        self.ptsTubZ = None
        self.ptsTopX = None
        self.ptsTopY = None
        self.ptsTopZ = None
        self.ptsBotX = None
        self.ptsBotY = None
        self.ptsBotZ = None

    def posrandomise(self, std_pos):
        """
        Takes as input a Cylinder and updates it with a random position, parameter tau
        :param cyl: geometry.Cylinder
        :param std_pos: for generation of random number
        :return: nothing (it updates the Cylinder)
        """
        ct = self.centre
        newx = ct[0] + rnd.gauss(0, std_pos)
        newy = ct[1] + rnd.gauss(0, std_pos)
        newz = ct[2] + rnd.gauss(0, std_pos)
        self.centre = np.array([newx, newy, newz])
        self.delPoints()

    def rotrandomise(self, std_pos):
        """
        Takes as input a Cylinder and updates it with a random axis (direction), parameter twist
        Randomness of rotation around x, around y and around z
        :param cyl: geometry.Cylinder
        :param std_pos: for generation of random number
        :return: nothing (it updates the Cylinder)
        """
        self.rotate(axis='x', theta=np.radians(rnd.gauss(0, std_pos)), point=self.centre)
        self.rotate(axis='y', theta=np.radians(rnd.gauss(0, std_pos)), point=self.centre)
        self.rotate(axis='z', theta=np.radians(rnd.gauss(0, std_pos)), point=self.centre)
        self.delPoints()

    def lengthrandomise(self, std_rodlength):
        """
        Takes as input a Cylinder and updates it with a random length, parameter tau
        :param cyl: geometry.Cylinder
        :param std_rodlength: for generation of random number
        :return: nothing (it updates the Cylinder)
        """
        self.length = rnd.gauss(self.length, std_rodlength)
        self.delPoints()

    def rotate(self, axis='z', theta=0, point=np.array([0, 0, 0])):
        """
        Rotate the Cylinder around an axis (direction) from an angle theta_rad, and the axis has a position
        :param axis: numpy array, or string that can be 'x', 'y' or 'z'
        :param theta: radians
        :param point: numpy array
        :return:
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
        if norm(axis) == 0:
            raise Exception('Invalid axis. Axis can not be (0, 0, 0).')
        else:
            # Translate everything from point to origin
            old_dir = self.direction
            old_ctr = self.centre
            old_ctr = old_ctr - point
            # Perform the rotation around the axis
            # Matrix here: https://en.wikipedia.org/wiki/Rotation_matrix
            axis = axis / norm(axis)
            ux = axis[0]
            uy = axis[1]
            uz = axis[2]
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            r = np.array([[costheta+(ux**2)*(1-costheta), ux*uy*(1-costheta)-uz*sintheta, ux*uz*(1-costheta)+uy*sintheta],
                          [uy*ux*(1-costheta)+uz*sintheta, costheta+(uy**2)*(1-costheta), uy*uz*(1-costheta)-ux*sintheta],
                          [uz*ux*(1-costheta)-uy*sintheta, uz*uy*(1-costheta)+ux*sintheta, costheta+(uz**2)*(1-costheta)]])
            # Translate everything back from origin to point
            new_dir = old_dir.dot(r)
            new_ctr = old_ctr.dot(r) + point
            # Save
            self.direction = new_dir/norm(new_dir)
            self.centre = new_ctr
            # Delete the points because the cylinder has been updated
            self.delPoints()

    def getPoints(self, resolution=10):
        """
        Calculates all the points to represent the Cylinder in 3D
        :param resolution: integer that defines the mesh size
        From here:
        https://stackoverflow.com/questions/39822480/plotting-a-solid-cylinder-centered-on-a-plane-in-matplotlib
        """
        # Make non-collinear unit vector
        vec_nc = np.array([rnd.random(), rnd.random(), rnd.random()])
        while norm(np.cross(self.direction, vec_nc)) == 0:
            vec_nc = np.array([rnd.random(), rnd.random(), rnd.random()])
        vec_nc = vec_nc / norm(vec_nc)
        # Make unit vectors perpendicular to direction
        vec1 = np.cross(self.direction, vec_nc)
        vec1 = vec1 / norm(vec1)
        # Make third vector of the set
        vec2 = np.cross(self.direction, vec1)
        # Mesh the surface
        pts_length = np.linspace(0, self.length, 2)
        pts_theta = np.linspace(0, 2 * np.pi, resolution)
        pts_radius = np.linspace(0, self.radius, 2)
        mesh_length, mesh_thetal = np.meshgrid(pts_length, pts_theta)
        mesh_r, mesh_thetar = np.meshgrid(pts_radius, pts_theta)
        # Find the bottom point
        p0 = self.centre - 0.5 * self.length * self.direction
        # Generate coordinates for the tube
        # Bottom-center point + take all perpendicular directions from the axis and plot on a circle perpendicular to the axis
        X, Y, Z = [p0[i] + self.direction[i] * pts_length + self.radius * np.sin(mesh_thetal) * vec1[i] + self.radius * np.cos(mesh_thetal) * vec2[i] for i in [0, 1, 2]]
        # Generate coordinates for the bottom
        X2, Y2, Z2 = [p0[i] + mesh_r[i] * np.sin(mesh_thetar) * vec1[i] + mesh_r[i] * np.cos(mesh_thetar) * vec2[i] for i in [0, 1, 2]]
        # Generate coordinates for the top
        X3, Y3, Z3 = [p0[i] + self.direction[i] * self.length + mesh_r[i] * np.sin(mesh_thetar) * vec1[i] + mesh_r[i] * np.cos(mesh_thetar) * vec2[i] for i in [0, 1, 2]]
        # Store points
        self.ptsTubX = X
        self.ptsTubY = Y
        self.ptsTubZ = Z
        self.ptsBotX = X2
        self.ptsBotY = Y2
        self.ptsBotZ = Z2
        self.ptsTopX = X3
        self.ptsTopY = Y3
        self.ptsTopZ = Z3

    def plot(self, ax, resolution=10):
        """
        Plots the Cylinder in 3D in a plot
        :param ax: axis from Pyplot (in 3D)
        :return:
        """
        self.getPoints(resolution)
        ax.plot_surface(self.ptsTubX, self.ptsTubY, self.ptsTubZ, color=self.color)
        ax.plot_surface(self.ptsTopX, self.ptsTopY, self.ptsTopZ, color='green')  # green
        ax.plot_surface(self.ptsBotX, self.ptsBotY, self.ptsBotZ, color='red')  # red

    def copy(self, cyl):
        """
        Copies all info from one Cylinder to another
        The empty Cylinder must be created first
        :param cyl: the Cylinder to duplicate
        :return: nothing, updates the Cylinder
        """
        self.centre = cyl.centre
        self.direction = cyl.direction
        self.length = cyl.length
        self.radius = cyl.radius
        self.color = cyl.colour
        self.ptsTubX = cyl.ptsTubX
        self.ptsTubY = cyl.ptsTubY
        self.ptsTubZ = cyl.ptsTubZ
        self.ptsTopX = cyl.ptsTopX
        self.ptsTopY = cyl.ptsTopY
        self.ptsTopZ = cyl.ptsTopZ
        self.ptsBotX = cyl.ptsBotX
        self.ptsBotY = cyl.ptsBotY
        self.ptsBotZ = cyl.ptsBotZ


class Arrow3D(FancyArrowPatch):
    # https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


