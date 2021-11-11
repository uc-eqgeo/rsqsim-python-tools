from typing import Union, List

import numpy as np
import math
from numba import njit
from pyproj import Transformer
from shapely.geometry import Polygon
from tde.tde import calc_tri_displacements


transformer_utm2nztm = Transformer.from_crs(32759, 2193, always_xy=True)
transformer_nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)

@njit(cache=True)
def norm_3d(a):
    """
    Calculates the 2-norm of a 3-dimensional vector.
    """
    return math.sqrt(np.sum(a**2))

@njit(cache=True)
def cross_3d(a, b):
    """
    Calculates cross product of two 3-dimensional vectors.
    """
    x = ((a[1] * b[2]) - (a[2] * b[1]))
    y = ((a[2] * b[0]) - (a[0] * b[2]))
    z = ((a[0] * b[1]) - (a[1] * b[0]))
    return np.array([x, y, z])


def normalize_bearing(bearing: Union[float, int]):
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing








class RsqSimGenericPatch:
    def __init__(self, segment, patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None, rake: float = None, total_slip: float = None):
        self._patch_number = None
        self._vertices = None
        self._dip_slip = None
        self._strike_slip = None
        self._rake = None

        self.segment = segment
        self.patch_number = patch_number
        self.dip_slip = dip_slip
        self.strike_slip = strike_slip
        self.rake = rake

    # Patch number is zero if not specified
    @property
    def patch_number(self):
        return self._patch_number

    @patch_number.setter
    def patch_number(self, patch_number: np.integer):
        assert isinstance(patch_number, (np.integer, int)), "Patch number must be an integer"
        assert patch_number >= 0, "Must be greater than zero!"
        self._patch_number = patch_number

    # I've written dip slip and strike slip as properties, in case we want to implement checks on values later
    @property
    def dip_slip(self):
        return self._dip_slip

    @dip_slip.setter
    def dip_slip(self, slip: float):
        self._dip_slip = slip

    @property
    def strike_slip(self):
        return self._strike_slip

    @strike_slip.setter
    def strike_slip(self, slip: float):
        self._strike_slip = slip

    @property
    def rake(self):
        return self._rake

    @rake.setter
    def rake(self, rake_i: float):
        if rake_i is not None:
            self._rake = normalize_bearing(rake_i)
        else:
            self._rake = None

    @property
    def vertices(self):
        return self._vertices

    @property
    def total_slip(self):
        ss = self.strike_slip if self.strike_slip is not None else 0.
        ds = self.dip_slip if self.dip_slip is not None else 0.
        return np.linalg.norm(np.array([ss, ds]))


class RsqSimTriangularPatch(RsqSimGenericPatch):
    """
    class to store information on an individual triangular patch of a fault
    """

    def __init__(self, segment, vertices: Union[list, np.ndarray, tuple], patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None, patch_data: Union[list, np.ndarray, tuple] = None,
                 rake: float = None, total_slip: float = None):

        super(RsqSimTriangularPatch, self).__init__(segment=segment, patch_number=patch_number,
                                                    dip_slip=dip_slip, strike_slip=strike_slip, rake=rake)
        self.vertices = vertices
        if patch_data is not None:
            self._normal_vector = patch_data[0]
            self._down_dip_vector = patch_data[1]
            self._dip = patch_data[2]
            self._along_strike_vector = patch_data[3]
            self._centre = patch_data[4]
            self._area = patch_data[5]
        else:
            self._normal_vector = RsqSimTriangularPatch.calculate_normal_vector(self.vertices)
            self._down_dip_vector = RsqSimTriangularPatch.calculate_down_dip_vector(self.normal_vector)
            self._dip = RsqSimTriangularPatch.calculate_dip(self.down_dip_vector)
            self._along_strike_vector = RsqSimTriangularPatch.calculate_along_strike_vector(self.normal_vector, self.down_dip_vector)
            self._centre = RsqSimTriangularPatch.calculate_centre(self.vertices)
            self._area = RsqSimTriangularPatch.calculate_area(self.vertices)

        if total_slip is not None:
            self.strike_slip = total_slip * np.cos(np.radians(self.rake))
            self.dip_slip = total_slip * np.sin(np.radians(self.rake))

    @RsqSimGenericPatch.vertices.setter
    def vertices(self, vertices: Union[list, np.ndarray, tuple]):
        try:
            vertex_array = np.array(vertices, dtype=np.float64)
        except ValueError:
            raise ValueError("Error parsing vertices for {}, patch {:d}".format(self.segment.name, self.patch_number))

        assert vertex_array.ndim == 2, "2D array expected"
        # Check that at least 3 vertices supplied
        assert vertex_array.shape[0] >= 3, "At least 3 vertices (rows in array) expected"
        assert vertex_array.shape[1] == 3, "Three coordinates (x,y,z expected for each vertex"

        if vertex_array.shape[0] > 4:
            print("{}, patch {:d}: more patches than expected".format(self.segment.name, self.patch_number))
            print("Taking first 3 vertices...")

        self._vertices = vertices[:3, :]

    @property
    def vertices_lonlat(self):

        return np.array(transformer_nztm2wgs.transform(*self.vertices.T)).T

    @staticmethod
    @njit(cache=True)
    def calculate_normal_vector(vertices):
        a = vertices[1] - vertices[0]
        b = vertices[1] - vertices[2]
        cross_a_b = cross_3d(a, b)
        # Ensure that normal always points up, normalize to give unit vector
        if cross_a_b[-1] < 0:
            unit_cross = -1 * cross_a_b / norm_3d(cross_a_b)
        else:
            unit_cross = cross_a_b / norm_3d(cross_a_b)
        return unit_cross

    @property
    def normal_vector(self):
        return self._normal_vector

    @property
    def down_dip_vector(self):
        return self._down_dip_vector

    @staticmethod
    @njit(cache=True)
    def calculate_down_dip_vector(normal_vector):
        dx, dy, dz = normal_vector
        if dz == 0:
            return np.array([0., 0., np.nan])
        else:
            dd_vec = np.array([dx, dy, -1 / dz])
            return dd_vec / norm_3d(dd_vec)

    @property
    def dip(self):
        return self._dip

    @staticmethod
    @njit(cache=True)
    def calculate_dip(down_dip_vector):
        if np.isnan(down_dip_vector[-1]):
            return np.nan
        else:
            horizontal = norm_3d(down_dip_vector[:-1])
            vertical = -1 * down_dip_vector[-1]
            return np.degrees(np.arctan(vertical / horizontal))

    @property
    def along_strike_vector(self):
        return self._along_strike_vector

    @property
    def strike(self):
        if self.along_strike_vector is not None:
            strike = 90. - np.degrees(np.arctan2(self.along_strike_vector[1], self.along_strike_vector[0]))
            return normalize_bearing(strike)
        else:
            return None

    @staticmethod
    @njit(cache=True)
    def calculate_along_strike_vector(normal_vector, down_dip_vector):
        return cross_3d(normal_vector, down_dip_vector)

    @property
    def centre(self):
        return self._centre

    @staticmethod
    @njit(cache=True)
    def calculate_centre(vertices):
        # np.mean(vertices, axis=0) does not have compile support
        return np.sum(vertices, axis=0) / len(vertices)

    @property
    def area(self):
        return self._area

    @staticmethod
    @njit(cache=True)
    def calculate_area(vertices):
        a = vertices[1] - vertices[0]
        b = vertices[1] - vertices[2]
        cross_a_b = cross_3d(a, b)
        norm = norm_3d(cross_a_b)
        area = 0.5 * norm
        return area

    def slip3d_to_ss_ds(self, x1_slip: Union[float, int], x2_slip: Union[float, int], x3_slip: Union[float, int]):
        sv_3d = np.array([x1_slip, x2_slip, x3_slip])
        ss = np.dot(self.along_strike_vector, sv_3d)
        ds = np.dot(-1 * self.down_dip_vector, sv_3d)

        return ds, ss

    def horizontal_sv_to_ds_ss(self, slipvec, magnitude: Union[float, int] = 1):
        """
        Program to perform the 'inverse' of slipvec.
        Arguments: strike, dip azimuth of slip vector (all degrees)
        Returns rake (degrees)
        """
        assert isinstance(slipvec, ())
        angle = self.strike - slipvec
        if angle < -180.:
            angle = 360. + angle
        elif angle > 180.:
            angle = angle - 360.

        if angle == 90.:
            strike_perp = magnitude
            strike_par = 0.
        elif angle == -90.:
            strike_perp = -1. * magnitude
            strike_par = 0
        else:
            strike_par = np.cos(np.radians(angle))
            strike_perp = np.cos(np.radians(angle)) / np.cos(np.radians(self.dip))
            normalizer = magnitude / np.linalg.norm(np.array([strike_par, strike_perp]))
            strike_par *= normalizer
            strike_perp *= normalizer

        return strike_perp, strike_par

    def as_polygon(self):
        return Polygon(self.vertices)

    def calculate_tsunami_greens_functions(self, x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray,
                                           grid_shape: tuple, poisson_ratio: float = 0.25):
        assert all([isinstance(a, np.ndarray) for a in [x_array, y_array]])
        assert x_array.shape == y_array.shape == z_array.shape
        assert x_array.ndim == 1

        assert all([a is not None for a in (self.dip_slip, self.strike_slip)])

        xv, yv, zv = [self.vertices.T[i] for i in range(3)]
        gf = calc_tri_displacements(x_array, y_array, z_array, xv, yv, -1. * zv,
                                       poisson_ratio, self.strike_slip, 0., self.dip_slip)

        vert_disp = np.array(gf["z"])
        vert_grid = vert_disp.reshape(grid_shape[1:])
        return vert_grid

    def calculate_3d_greens_functions(self, x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray = None,
                                      poisson_ratio: float = 0.25):
        assert all([isinstance(a, np.ndarray) for a in [x_array, y_array]])
        assert x_array.shape == y_array.shape
        assert x_array.ndim == 1
        if z_array is None:
            z_array = np.zeros(x_array.shape)
        else:
            assert z_array.shape == x_array.shape

        assert all([a is not None for a in (self.dip_slip, self.strike_slip)])

        xv, yv, zv = [self.vertices.T[i] for i in range(3)]
        gf = calc_tri_displacements(x_array, y_array, z_array, xv, yv, -1. * zv,
                                    poisson_ratio, self.strike_slip, 0., self.dip_slip)

        return gf