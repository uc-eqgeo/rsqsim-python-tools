from typing import Union

import xml.etree.ElementTree as ElemTree
from xml.dom import minidom

import numpy as np
import math
from numba import njit
from pyproj import Transformer
from shapely.geometry import Polygon


transformer_utm2nztm = Transformer.from_crs(32759, 2193, always_xy=True)
transformer_nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)
transformer_wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

anticlockwise90 = np.array([[0., -1., 0.],
                            [1., 0., 0.],
                            [0., 0., 1.]])


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


@njit(cache=True)
def unit_vector(vec1: np.ndarray, vec2: np.ndarray):
    return (vec2 - vec1) / norm_3d(vec2 - vec1)






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
        if self._strike_slip is not None:
            self.rake = self.calculate_rake(self._strike_slip, slip)
        self._dip_slip = slip

    @property
    def strike_slip(self):
        return self._strike_slip

    @strike_slip.setter
    def strike_slip(self, slip: float):
        if self.dip_slip is not None:
            self.rake = self.calculate_rake(slip, self.dip_slip)
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

    def set_slip_rake(self, total_slip: float, rake: float):
        self.rake = rake
        self._strike_slip = total_slip * np.cos(np.radians(rake))
        self._dip_slip = total_slip * np.sin(np.radians(rake))

    @staticmethod
    def calculate_rake(strike_slip: float, dip_slip: float):
        return normalize_bearing(np.degrees(np.arctan2(dip_slip, strike_slip)))



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
            self.set_slip_rake(total_slip, self.rake)

    @RsqSimGenericPatch.vertices.setter
    def vertices(self, vertices: Union[list, np.ndarray, tuple]):
        try:
            vertex_array = np.array(vertices, dtype=np.float64)
        except ValueError:
            raise ValueError("Error parsing vertices for {}, patch {:d}".format(self.segment.name, self.patch_number))

        assert vertex_array.ndim == 2, "2D array expected"
        # Check that at least 3 vertices supplied
        assert vertex_array.shape[0] >= 3, "At least 3 vertices (rows in array) expected"
        assert vertex_array.shape[1] == 3, "Three coordinates (x,y,z) expected for each vertex"

        if vertex_array.shape[0] > 4:
            print("{}, patch {:d}: more patches than expected".format(self.segment.name, self.patch_number))
            print("Taking first 3 vertices...")

        self._vertices = vertices[:3, :]

    @property
    def vertices_lonlat(self):

        return np.array(transformer_nztm2wgs.transform(*self.vertices.T)).T

    @property
    def centre_lonlat(self):

        return np.array(transformer_nztm2wgs.transform(*self.centre.T)).T

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
            dd_vec = np.array([0., 0., -1.])
        else:
            xy_mag = np.sqrt(dx ** 2 + dy ** 2)
            xy_scaling = abs(dz) / xy_mag
            dd_vec = np.array([dx * xy_scaling, dy * xy_scaling, -xy_mag])
        return dd_vec

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
            if horizontal < 1e-10 :
                return 90.
            else:
                vertical = -1 * down_dip_vector[-1]
                return np.degrees(np.arctan(vertical / horizontal))

    @property
    def along_strike_vector(self):
        return self._along_strike_vector

    @property
    def strike(self):
        if any([self.normal_vector is None, self.down_dip_vector is None, self.along_strike_vector is None]):
            self.calculate_normal_vector()
            self.calculate_down_dip_vector()
            self.calculate_along_strike_vector()
        strike = 90. - np.degrees(np.arctan2(self.along_strike_vector[1], self.along_strike_vector[0]))
        return normalize_bearing(strike)

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
        Function to perform the 'inverse' of slipvec.
        Requires strike and dip of patch to be set (degrees)
        Returns strike perpendicular & strike parallel components of rake vector and rake (degrees)

        Parameters
        ----------
        slipvec :  azimuth of slip vector (degrees)
        magnitude : magnitude of slip vector (results are normalised)
        """
        #assert isinstance(slipvec, ())
        # angle is the angle between the horizontal azimuth of the slip vector and the strike
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
            strike_perp = np.sin(np.radians(angle)) / np.cos(np.radians(self.dip))
            normalizer = magnitude / np.linalg.norm(np.array([strike_par, strike_perp]))
            strike_par *= normalizer
            strike_perp *= normalizer
            rake = np.rad2deg(np.arctan2(strike_perp,strike_par))

        return strike_perp, strike_par, rake

    def slip_vec_3d(self):
        strike_slip = self.strike_slip * self.along_strike_vector
        dip_slip = self.dip_slip * self.down_dip_vector * -1
        return strike_slip + dip_slip

    def rake_from_stress_tensor(self, sigma1: np.ndarray):
        """
        Calculate rake for motion in direction of shear stress resolved on to fault patch. Assumes sigma2=sigma3=0.
        Parameters
        ----------
        sigma1 : maximum principle stress as vector (might need to change this to include multiple principle stresses and/or different ways of writing the principal stresses) 
        
        """

        assert len(sigma1) == 3

        assert all([ a is not None for a in (self.along_strike_vector,self.normal_vector)])

        if norm_3d(self.normal_vector == 1.) :
            unit_norm=self.normal_vector
        else:
            unit_norm=self.normal_vector/norm_3d(self.normal_vector)

        normalStress=np.dot(sigma1,unit_norm)*unit_norm
        shearStress=sigma1 - normalStress

        self.rake = np.rad2deg(np.arccos(np.dot(shearStress,self.along_strike_vector)/(norm_3d(shearStress)*norm_3d(self.along_strike_vector))))

    @property
    def vertical_slip(self):
        return self.dip_slip * np.cos(np.radians(self.dip))





    def as_polygon(self):
        return Polygon(self.vertices)

class OpenQuakeRectangularPatch(RsqSimGenericPatch):
    def __init__(self, segment, patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None, rake: float = None):
        super(OpenQuakeRectangularPatch, self).__init__(segment, patch_number=patch_number, dip_slip=dip_slip,
                                                        strike_slip=strike_slip, rake=rake)

        self._top_left, self._top_right = (None,) * 2
        self._bottom_left, self._bottom_right = (None,) * 2

        self._along_strike_vector = None
        self._down_dip_vector = None

    @property
    def top_left(self):
        return self._top_left

    @property
    def top_right(self):
        return self._top_right

    @property
    def bottom_left(self):
        return self._bottom_left

    @property
    def bottom_right(self):
        return self._bottom_right

    @property
    def along_strike_vector(self):
        return unit_vector(self.top_left, self.top_right)

    @property
    def top_centre(self):
        return 0.5 * (self.top_left + self.top_right)

    @property
    def bottom_centre(self):
        return 0.5 * (self.bottom_left + self.bottom_right)

    @property
    def down_dip_vector(self):
        return unit_vector(self.top_centre, self.bottom_centre)

    @classmethod
    def from_polygon(cls, polygon: Polygon, segment=None, patch_number: int = 0,
                     dip_slip: float = None, strike_slip: float = None, rake: float = None,
                     wgs_to_nztm: bool = False):
        coords = np.array(polygon.exterior.coords)
        assert coords.shape == (5, 3)
        coords = coords[:-1]

        top_depth = max(coords[:, -1])
        bottom_depth = min(coords[:, -1])
        top1, top2 = coords[coords[:, -1] == top_depth]
        bot1, bot2 = coords[coords[:, -1] == bottom_depth]

        if wgs_to_nztm:
            top1 = np.array(transformer_wgs2nztm.transform(*top1))
            top2 = np.array(transformer_wgs2nztm.transform(*top2))
            bot1 = np.array(transformer_wgs2nztm.transform(*bot1))
            bot2 = np.array(transformer_wgs2nztm.transform(*bot2))

        patch = cls(segment, patch_number=patch_number, dip_slip=dip_slip,
                    strike_slip=strike_slip, rake=rake)

        # Check for vertical patch:
        if any([np.array_equal(top1[:-1], a[:-1]) for a in [bot1, bot2]]):
            patch._top_left = top1
            patch._top_right = top2
            patch._along_strike_vector = (top2 - top1) / np.linalg.norm(top2 - top1)
            patch._down_dip_vector = np.array([0., 0., -1.])
            if np.array_equal(top1[:-1], bot1[:-1]):
                patch._bottom_left = bot1
                patch._bottom_right = bot2
            else:
                patch._bottom_left = bot2
                patch._bottom_right = bot1

        else:
            # Try one way round, if it doesn't work try the other way

            cen_top = 0.5 * (top1 + top2)
            cen_bot = 0.5 * (bot1 + bot2)

            across_strike_vector = unit_vector(cen_top, np.array([cen_bot[0], cen_bot[1], cen_top[-1]]))
            along_strike_vector = np.matmul(anticlockwise90, across_strike_vector)

            if np.dot(along_strike_vector, unit_vector(top1, top2)) > 0:
                patch._top_left = top1
                patch._top_right = top2
            else:
                patch._top_left = top2
                patch._top_right = top1

            if np.dot(along_strike_vector, unit_vector(bot1, bot2)) > 0:
                patch._bottom_left = bot1
                patch._bottom_right = bot2
            else:
                patch._bottom_left = bot2
                patch._bottom_right = bot1

        return patch

    def to_oq_xml(self):
        plane_element = ElemTree.Element("planarSurface")

        corner_list = ["topLeft", "topRight", "bottomLeft", "bottomRight"]
        for label, corner in zip(corner_list, [self.top_left, self.top_right, self.bottom_left, self.bottom_right]):
            depth_km = -1.e-3 * corner[-1]
            lon, lat = transformer_nztm2wgs.transform(corner[0], corner[1])
            element_i = ElemTree.Element(label, attrib={"depth": f"{depth_km:.4f}",
                                                        "lat": f"{lat:.4f}",
                                                        "lon": f"{lon:.4f}"
                                                        })
            plane_element.append(element_i)

        return plane_element

















