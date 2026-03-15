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
    Calculate the Euclidean (2-norm) length of a 3-dimensional vector.

    Parameters
    ----------
    a : array-like of shape (3,)
        Input 3-component vector.

    Returns
    -------
    float
        Scalar Euclidean norm of ``a``.

    Notes
    -----
    Compiled with ``numba.njit`` for performance inside tight loops.
    """
    return math.sqrt(np.sum(a**2))


@njit(cache=True)
def cross_3d(a, b):
    """
    Calculate the cross product of two 3-dimensional vectors.

    Parameters
    ----------
    a : array-like of shape (3,)
        First input vector.
    b : array-like of shape (3,)
        Second input vector.

    Returns
    -------
    numpy.ndarray of shape (3,)
        Vector perpendicular to both ``a`` and ``b``, with magnitude
        ``|a| * |b| * sin(theta)``, following the right-hand rule.

    Notes
    -----
    Compiled with ``numba.njit`` for performance inside tight loops.
    """
    x = ((a[1] * b[2]) - (a[2] * b[1]))
    y = ((a[2] * b[0]) - (a[0] * b[2]))
    z = ((a[0] * b[1]) - (a[1] * b[0]))
    return np.array([x, y, z])


def normalize_bearing(bearing: Union[float, int]):
    """
    Wrap a bearing value into the half-open interval [0, 360).

    Parameters
    ----------
    bearing :
        Input bearing in degrees.  May be any real number.

    Returns
    -------
    float
        Equivalent bearing constrained to [0, 360).
    """
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing


@njit(cache=True)
def unit_vector(vec1: np.ndarray, vec2: np.ndarray):
    """
    Compute the unit vector pointing from ``vec1`` to ``vec2``.

    Parameters
    ----------
    vec1 : numpy.ndarray of shape (3,)
        Origin point (tail of the vector).
    vec2 : numpy.ndarray of shape (3,)
        Destination point (head of the vector).

    Returns
    -------
    numpy.ndarray of shape (3,)
        Normalised direction vector from ``vec1`` to ``vec2``.

    Notes
    -----
    Compiled with ``numba.njit`` for performance inside tight loops.
    """
    return (vec2 - vec1) / norm_3d(vec2 - vec1)


class RsqSimGenericPatch:
    """
    Base class representing a single fault patch in an RSQSim fault model.

    Stores patch identity, slip components, and the parent fault segment
    reference.  Concrete subclasses add geometry (triangular, rectangular,
    etc.).

    Parameters
    ----------
    segment :
        Parent fault segment object that owns this patch.
    patch_number : int, optional
        Zero-based integer index identifying the patch within its segment.
        Defaults to 0.
    dip_slip : float, optional
        Dip-slip component of displacement in metres.  Positive values
        indicate reverse/thrust motion.
    strike_slip : float, optional
        Strike-slip component of displacement in metres.  Positive values
        indicate right-lateral motion.
    rake : float, optional
        Rake angle in degrees, measured from the along-strike direction
        following the Aki & Richards convention.  Stored normalised to
        [0, 360).
    total_slip : float, optional
        Total slip magnitude in metres (unused by the base class but
        accepted for interface consistency).

    Attributes
    ----------
    segment :
        Reference to the parent fault segment.
    patch_number : int
        Non-negative integer patch index.
    dip_slip : float or None
        Dip-slip component in metres.
    strike_slip : float or None
        Strike-slip component in metres.
    rake : float or None
        Rake angle in degrees, normalised to [0, 360).
    """

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
        """
        Non-negative integer index identifying this patch within its segment.
        """
        return self._patch_number

    @patch_number.setter
    def patch_number(self, patch_number: np.integer):
        """
        Set the patch number after validating it is a non-negative integer.

        Parameters
        ----------
        patch_number :
            Non-negative integer patch index.

        Raises
        ------
        AssertionError
            If ``patch_number`` is not an integer type or is negative.
        """
        assert isinstance(patch_number, (np.integer, int)), "Patch number must be an integer"
        assert patch_number >= 0, "Must be greater than zero!"
        self._patch_number = patch_number

    # I've written dip slip and strike slip as properties, in case we want to implement checks on values later
    @property
    def dip_slip(self):
        """
        Dip-slip displacement component in metres.

        Setting this value also recomputes the rake if the strike-slip
        component is already defined.
        """
        return self._dip_slip

    @dip_slip.setter
    def dip_slip(self, slip: float):
        """
        Assign the dip-slip component and update rake if strike-slip is set.

        Parameters
        ----------
        slip :
            Dip-slip displacement in metres.
        """
        if self._strike_slip is not None:
            self.rake = self.calculate_rake(self._strike_slip, slip)
        self._dip_slip = slip

    @property
    def strike_slip(self):
        """
        Strike-slip displacement component in metres.

        Setting this value also recomputes the rake if the dip-slip
        component is already defined.
        """
        return self._strike_slip

    @strike_slip.setter
    def strike_slip(self, slip: float):
        """
        Assign the strike-slip component and update rake if dip-slip is set.

        Parameters
        ----------
        slip :
            Strike-slip displacement in metres.
        """
        if self.dip_slip is not None:
            self.rake = self.calculate_rake(slip, self.dip_slip)
        self._strike_slip = slip

    @property
    def rake(self):
        """
        Rake angle in degrees, normalised to the interval [0, 360).

        Setting this value calls :func:`normalize_bearing` automatically.
        """
        return self._rake

    @rake.setter
    def rake(self, rake_i: float):
        """
        Assign the rake angle, normalising it to [0, 360).

        Parameters
        ----------
        rake_i :
            Rake angle in degrees.  Pass ``None`` to clear the rake.
        """
        if rake_i is not None:
            self._rake = normalize_bearing(rake_i)
        else:
            self._rake = None

    @property
    def vertices(self):
        """
        Array of patch corner vertices in NZTM coordinates (metres).

        Returns ``None`` until populated by a subclass setter.
        """
        return self._vertices

    @property
    def total_slip(self):
        """
        Euclidean magnitude of the slip vector in metres.

        Computed from ``strike_slip`` and ``dip_slip``; treats ``None``
        components as zero.
        """
        ss = self.strike_slip if self.strike_slip is not None else 0.
        ds = self.dip_slip if self.dip_slip is not None else 0.
        return np.linalg.norm(np.array([ss, ds]))

    def set_slip_rake(self, total_slip: float, rake: float):
        """
        Decompose a total slip magnitude and rake into strike-slip and dip-slip.

        Sets ``strike_slip``, ``dip_slip``, and ``rake`` on the patch
        directly without triggering the cross-update logic in the individual
        property setters.

        Parameters
        ----------
        total_slip :
            Total slip magnitude in metres.
        rake :
            Rake angle in degrees (stored normalised to [0, 360)).
        """
        self.rake = rake
        self._strike_slip = total_slip * np.cos(np.radians(rake))
        self._dip_slip = total_slip * np.sin(np.radians(rake))

    @staticmethod
    def calculate_rake(strike_slip: float, dip_slip: float):
        """
        Compute rake from strike-slip and dip-slip components.

        Parameters
        ----------
        strike_slip :
            Strike-slip displacement component in metres.
        dip_slip :
            Dip-slip displacement component in metres.

        Returns
        -------
        float
            Rake angle in degrees, normalised to [0, 360) via
            :func:`normalize_bearing`.
        """
        return normalize_bearing(np.degrees(np.arctan2(dip_slip, strike_slip)))


class RsqSimTriangularPatch(RsqSimGenericPatch):
    """
    A single triangular fault patch for use in RSQSim fault models.

    Stores the three corner vertices and derives geometric quantities
    (normal vector, down-dip vector, along-strike vector, dip, centre,
    area) either from supplied ``patch_data`` or by computing them from
    the vertices.

    Parameters
    ----------
    segment :
        Parent fault segment object that owns this patch.
    vertices : list or numpy.ndarray or tuple
        Array-like of shape (3, 3) giving the (x, y, z) coordinates of
        the triangle corners in NZTM (metres).  If more than three rows
        are supplied only the first three are used.
    patch_number : int, optional
        Zero-based integer index identifying the patch.  Defaults to 0.
    dip_slip : float, optional
        Dip-slip component of displacement in metres.
    strike_slip : float, optional
        Strike-slip component of displacement in metres.
    patch_data : list or numpy.ndarray or tuple, optional
        Pre-computed geometry sequence of the form
        ``[normal_vector, down_dip_vector, dip, along_strike_vector,
        centre, area]``.  When provided the geometric calculations are
        skipped.
    rake : float, optional
        Rake angle in degrees.
    total_slip : float, optional
        Total slip magnitude in metres.  When provided, ``set_slip_rake``
        is called using this value and the current rake.

    Attributes
    ----------
    segment :
        Reference to the parent fault segment.
    patch_number : int
        Non-negative integer patch index.
    vertices : numpy.ndarray of shape (3, 3)
        Triangle corner coordinates in NZTM (metres).
    normal_vector : numpy.ndarray of shape (3,)
        Unit normal vector of the patch pointing upward.
    down_dip_vector : numpy.ndarray of shape (3,)
        Vector pointing in the steepest down-dip direction.
    along_strike_vector : numpy.ndarray of shape (3,)
        Vector pointing along strike.
    dip : float
        Dip angle in degrees, measured from horizontal.
    centre : numpy.ndarray of shape (3,)
        Centroid of the triangle in NZTM (metres).
    area : float
        Area of the triangular patch in square metres.
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
        """
        Validate and store the triangle corner vertices.

        Accepts an array-like with at least three rows of three (x, y, z)
        coordinates in NZTM (metres).  Only the first three rows are
        retained if more are provided.

        Parameters
        ----------
        vertices :
            Array-like of shape (N, 3) where N >= 3.  Each row is an
            (x, y, z) coordinate in NZTM (metres).

        Raises
        ------
        ValueError
            If ``vertices`` cannot be converted to a float64 NumPy array.
        AssertionError
            If the array is not 2-D, has fewer than 3 rows, or does not
            have exactly 3 columns.
        """
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
        """
        Triangle corner coordinates transformed to WGS84 longitude/latitude.

        Returns
        -------
        numpy.ndarray of shape (3, 3)
            Columns are (longitude, latitude, depth) for each of the three
            corners, in degrees (WGS84) and metres depth.
        """
        return np.array(transformer_nztm2wgs.transform(*self.vertices.T)).T

    @property
    def centre_lonlat(self):
        """
        Patch centroid transformed to WGS84 longitude/latitude.

        Returns
        -------
        numpy.ndarray of shape (3,)
            (longitude, latitude, depth) of the patch centroid, in degrees
            (WGS84) and metres depth.
        """
        return np.array(transformer_nztm2wgs.transform(*self.centre.T)).T

    @staticmethod
    @njit(cache=True)
    def calculate_normal_vector(vertices):
        """
        Compute the upward-pointing unit normal vector of a triangular patch.

        Uses the cross product of two edge vectors.  If the raw cross
        product points downward its sign is flipped so that the normal
        always has a positive z-component.

        Parameters
        ----------
        vertices : numpy.ndarray of shape (3, 3)
            Triangle corner coordinates (x, y, z) in any consistent
            Cartesian coordinate system.

        Returns
        -------
        numpy.ndarray of shape (3,)
            Unit normal vector with non-negative z-component.

        Notes
        -----
        Compiled with ``numba.njit`` for performance.
        """
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
        """
        Upward-pointing unit normal vector of the triangular patch.
        """
        return self._normal_vector

    @property
    def down_dip_vector(self):
        """
        Vector pointing in the steepest down-dip direction of the patch.
        """
        return self._down_dip_vector

    @staticmethod
    @njit(cache=True)
    def calculate_down_dip_vector(normal_vector):
        """
        Derive the down-dip vector from a patch normal vector.

        The down-dip direction is the steepest descent direction on the
        fault plane.  For a vertical patch (zero z-component in the
        normal) it defaults to straight down (0, 0, -1).

        Parameters
        ----------
        normal_vector : numpy.ndarray of shape (3,)
            Upward-pointing unit normal of the fault patch.

        Returns
        -------
        numpy.ndarray of shape (3,)
            Down-dip vector whose z-component is ``-xy_mag`` and whose
            horizontal components are scaled by ``|dz| / xy_mag``.

        Notes
        -----
        Compiled with ``numba.njit`` for performance.
        """
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
        """
        Dip angle of the patch in degrees, measured from horizontal.

        Returns 90 for vertical patches and ``numpy.nan`` if the down-dip
        vector contains NaN values.
        """
        return self._dip

    @staticmethod
    @njit(cache=True)
    def calculate_dip(down_dip_vector):
        """
        Calculate the dip angle in degrees from a down-dip vector.

        Parameters
        ----------
        down_dip_vector : numpy.ndarray of shape (3,)
            Down-dip direction vector (need not be a unit vector).

        Returns
        -------
        float
            Dip angle in degrees measured from horizontal.  Returns
            ``numpy.nan`` if ``down_dip_vector[-1]`` is NaN, or 90.0
            when the horizontal component magnitude is less than 1e-10
            (effectively vertical patch).

        Notes
        -----
        Compiled with ``numba.njit`` for performance.
        """
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
        """
        Vector pointing along the strike direction of the patch.
        """
        return self._along_strike_vector

    @property
    def strike(self):
        """
        Strike bearing of the patch in degrees, normalised to [0, 360).

        Derived from the along-strike vector using the standard geographic
        convention (clockwise from north).  Recomputes normal, down-dip,
        and along-strike vectors if any are ``None``.
        """
        if any([self.normal_vector is None, self.down_dip_vector is None, self.along_strike_vector is None]):
            self.calculate_normal_vector()
            self.calculate_down_dip_vector()
            self.calculate_along_strike_vector()
        strike = 90. - np.degrees(np.arctan2(self.along_strike_vector[1], self.along_strike_vector[0]))
        return normalize_bearing(strike)

    @staticmethod
    @njit(cache=True)
    def calculate_along_strike_vector(normal_vector, down_dip_vector):
        """
        Compute the along-strike vector as the cross product of the normal
        and down-dip vectors.

        Parameters
        ----------
        normal_vector : numpy.ndarray of shape (3,)
            Upward-pointing unit normal of the fault patch.
        down_dip_vector : numpy.ndarray of shape (3,)
            Down-dip direction vector of the fault patch.

        Returns
        -------
        numpy.ndarray of shape (3,)
            Along-strike vector (normal cross down-dip, right-hand rule).

        Notes
        -----
        Compiled with ``numba.njit`` for performance.
        """
        return cross_3d(normal_vector, down_dip_vector)

    @property
    def centre(self):
        """
        Centroid of the triangular patch in NZTM coordinates (metres).
        """
        return self._centre

    @staticmethod
    @njit(cache=True)
    def calculate_centre(vertices):
        """
        Calculate the centroid of a triangular patch.

        Parameters
        ----------
        vertices : numpy.ndarray of shape (3, 3)
            Triangle corner coordinates (x, y, z) in NZTM (metres).

        Returns
        -------
        numpy.ndarray of shape (3,)
            Mean of the three corner coordinates, i.e. the centroid.

        Notes
        -----
        ``numpy.mean`` is not supported by Numba's njit compiler, so the
        centroid is computed as ``sum / len`` directly.  Compiled with
        ``numba.njit`` for performance.
        """
        # np.mean(vertices, axis=0) does not have compile support
        return np.sum(vertices, axis=0) / len(vertices)

    @property
    def area(self):
        """
        Area of the triangular patch in square metres.
        """
        return self._area

    @staticmethod
    @njit(cache=True)
    def calculate_area(vertices):
        """
        Calculate the area of a triangular patch from its three vertices.

        Uses the cross-product formula: area = 0.5 * |a x b|, where ``a``
        and ``b`` are two edge vectors meeting at the same vertex.

        Parameters
        ----------
        vertices : numpy.ndarray of shape (3, 3)
            Triangle corner coordinates (x, y, z) in NZTM (metres).

        Returns
        -------
        float
            Area of the triangle in square metres.

        Notes
        -----
        Compiled with ``numba.njit`` for performance.
        """
        a = vertices[1] - vertices[0]
        b = vertices[1] - vertices[2]
        cross_a_b = cross_3d(a, b)
        norm = norm_3d(cross_a_b)
        area = 0.5 * norm
        return area

    def slip3d_to_ss_ds(self, x1_slip: Union[float, int], x2_slip: Union[float, int], x3_slip: Union[float, int]):
        """
        Decompose a 3-D slip vector into strike-slip and dip-slip components.

        Projects the supplied Cartesian slip vector onto the along-strike
        and (negated) down-dip directions of this patch.

        Parameters
        ----------
        x1_slip :
            Slip component in the x1 (easting) direction in metres.
        x2_slip :
            Slip component in the x2 (northing) direction in metres.
        x3_slip :
            Slip component in the x3 (vertical) direction in metres.

        Returns
        -------
        ds : float
            Dip-slip component in metres (projection onto the up-dip
            direction).
        ss : float
            Strike-slip component in metres (projection onto the
            along-strike direction).
        """
        sv_3d = np.array([x1_slip, x2_slip, x3_slip])
        ss = np.dot(self.along_strike_vector, sv_3d)
        ds = np.dot(-1 * self.down_dip_vector, sv_3d)

        return ds, ss

    def horizontal_sv_to_ds_ss(self, slipvec, magnitude: Union[float, int] = 1):
        """
        Convert a horizontal slip-vector azimuth to dip-slip and strike-slip components.

        Parameters
        ----------
        slipvec :
            Azimuth of the slip vector measured clockwise from north,
            in degrees.
        magnitude :
            Desired total magnitude of the output slip components.
            Results are normalised to this value.  Defaults to 1.

        Returns
        -------
        strike_perp : float
            Strike-perpendicular (dip-slip proxy) component, normalised
            to ``magnitude``.
        strike_par : float
            Strike-parallel (strike-slip proxy) component, normalised
            to ``magnitude``.
        rake : float
            Rake angle in degrees derived from
            ``arctan2(strike_perp, strike_par)``.

        Notes
        -----
        The angle between the patch strike and the input azimuth is used
        to project the horizontal slip direction onto the fault-plane
        axes, accounting for the patch dip when resolving the
        strike-perpendicular component.
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
        """
        Compute the 3-D slip vector for this patch.

        Combines the strike-slip and dip-slip components with their
        respective direction vectors to produce the full 3-D displacement
        vector in NZTM Cartesian coordinates (metres).

        Returns
        -------
        numpy.ndarray of shape (3,)
            3-D slip vector in NZTM (metres).
        """
        strike_slip = self.strike_slip * self.along_strike_vector
        dip_slip = self.dip_slip * self.down_dip_vector * -1
        return strike_slip + dip_slip

    def rake_from_stress_tensor(self, sigma1: np.ndarray):
        """
        Set the rake to align with the shear stress resolved onto the patch.

        Resolves the principal stress vector ``sigma1`` onto the fault
        plane, extracts the shear-stress component, and sets the patch
        rake to the angle between that shear stress and the along-strike
        direction.  Assumes sigma2 = sigma3 = 0.

        Parameters
        ----------
        sigma1 : numpy.ndarray of shape (3,)
            Maximum principal stress vector (direction and relative
            magnitude).

        Raises
        ------
        AssertionError
            If ``sigma1`` does not have exactly 3 elements, or if either
            ``along_strike_vector`` or ``normal_vector`` is ``None``.

        Notes
        -----
        Only the direction of ``sigma1`` influences the resulting rake;
        sigma2 and sigma3 are not accounted for.
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
        """
        Vertical component of the dip-slip displacement in metres.

        Computed as ``dip_slip * cos(dip)``, where dip is in degrees.
        Positive when the hanging wall moves upward relative to the
        footwall (reverse/thrust sense).
        """
        return self.dip_slip * np.cos(np.radians(self.dip))


    def as_polygon(self):
        """
        Return the triangular patch as a Shapely Polygon.

        Returns
        -------
        shapely.geometry.Polygon
            Polygon whose exterior ring is defined by the three patch
            vertices.
        """
        return Polygon(self.vertices)

class OpenQuakeRectangularPatch(RsqSimGenericPatch):
    """
    A rectangular fault patch suitable for use with OpenQuake engine inputs.

    Stores the four corner points of a planar rectangular fault surface
    and derives geometric direction vectors from them.  Provides a class
    method to construct a patch from a Shapely ``Polygon`` and an export
    method for writing OpenQuake-compatible XML.

    Parameters
    ----------
    segment :
        Parent fault segment object that owns this patch.
    patch_number : int, optional
        Zero-based integer index identifying the patch.  Defaults to 0.
    dip_slip : float, optional
        Dip-slip component of displacement in metres.
    strike_slip : float, optional
        Strike-slip component of displacement in metres.
    rake : float, optional
        Rake angle in degrees.

    Attributes
    ----------
    segment :
        Reference to the parent fault segment.
    patch_number : int
        Non-negative integer patch index.
    top_left : numpy.ndarray of shape (3,) or None
        Top-left corner in NZTM (metres); populated after construction.
    top_right : numpy.ndarray of shape (3,) or None
        Top-right corner in NZTM (metres); populated after construction.
    bottom_left : numpy.ndarray of shape (3,) or None
        Bottom-left corner in NZTM (metres); populated after construction.
    bottom_right : numpy.ndarray of shape (3,) or None
        Bottom-right corner in NZTM (metres); populated after construction.
    """

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
        """
        Top-left corner of the rectangular patch in NZTM (metres).
        """
        return self._top_left

    @property
    def top_right(self):
        """
        Top-right corner of the rectangular patch in NZTM (metres).
        """
        return self._top_right

    @property
    def bottom_left(self):
        """
        Bottom-left corner of the rectangular patch in NZTM (metres).
        """
        return self._bottom_left

    @property
    def bottom_right(self):
        """
        Bottom-right corner of the rectangular patch in NZTM (metres).
        """
        return self._bottom_right

    @property
    def along_strike_vector(self):
        """
        Unit vector pointing along strike, from top-left to top-right corner.
        """
        return unit_vector(self.top_left, self.top_right)

    @property
    def top_centre(self):
        """
        Midpoint of the top edge of the rectangular patch in NZTM (metres).
        """
        return 0.5 * (self.top_left + self.top_right)

    @property
    def bottom_centre(self):
        """
        Midpoint of the bottom edge of the rectangular patch in NZTM (metres).
        """
        return 0.5 * (self.bottom_left + self.bottom_right)

    @property
    def down_dip_vector(self):
        """
        Unit vector pointing down-dip, from top-centre to bottom-centre.
        """
        return unit_vector(self.top_centre, self.bottom_centre)

    @classmethod
    def from_polygon(cls, polygon: Polygon, segment=None, patch_number: int = 0,
                     dip_slip: float = None, strike_slip: float = None, rake: float = None,
                     wgs_to_nztm: bool = False):
        """
        Construct an ``OpenQuakeRectangularPatch`` from a Shapely Polygon.

        Parses the four corners of a rectangular fault-surface polygon,
        identifies which corners belong to the top and bottom edges based
        on depth (z-coordinate), and assigns them as top-left, top-right,
        bottom-left, and bottom-right using the along-strike direction to
        determine left/right orientation.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            Closed polygon with exactly 5 exterior coordinates (4 unique
            corners plus the closing repeat of the first).  Each
            coordinate must have three values (x, y, z); z is depth in
            metres using a negative-downward convention.
        segment :
            Parent fault segment object.  Defaults to ``None``.
        patch_number : int, optional
            Zero-based patch index.  Defaults to 0.
        dip_slip : float, optional
            Dip-slip displacement in metres.
        strike_slip : float, optional
            Strike-slip displacement in metres.
        rake : float, optional
            Rake angle in degrees.
        wgs_to_nztm : bool, optional
            If ``True``, the polygon coordinates are assumed to be in
            WGS84 (longitude, latitude, depth) and are reprojected to
            NZTM before assigning corners.  Defaults to ``False``.

        Returns
        -------
        OpenQuakeRectangularPatch
            New patch instance with all four corner coordinates set.

        Raises
        ------
        AssertionError
            If the polygon exterior does not have exactly 5 coordinate
            rows (i.e. 4 unique corners).

        Notes
        -----
        For vertical patches (where a top corner shares its horizontal
        position with a bottom corner) the down-dip vector is set to
        ``[0, 0, -1]`` and the along-strike vector is computed directly
        from the top edge.  For non-vertical patches an across-strike
        vector is projected horizontally and rotated 90 degrees
        anti-clockwise using the module-level ``anticlockwise90`` matrix
        to determine the along-strike direction, which in turn determines
        the left/right assignment of corner pairs.
        """
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
        """
        Serialise the patch as an OpenQuake ``planarSurface`` XML element.

        Converts the four corner coordinates from NZTM (metres) to WGS84
        longitude/latitude and depth in kilometres, then builds an
        ``xml.etree.ElementTree.Element`` with child elements for each
        corner.

        Returns
        -------
        xml.etree.ElementTree.Element
            ``<planarSurface>`` element containing four child elements
            named ``topLeft``, ``topRight``, ``bottomLeft``, and
            ``bottomRight``, each carrying ``depth`` (km, 4 d.p.),
            ``lat`` (degrees, 4 d.p.), and ``lon`` (degrees, 4 d.p.)
            attributes.

        Notes
        -----
        Depth is stored as a positive value in kilometres following the
        OpenQuake NRML convention, converted from the negative-downward
        metres used internally via ``depth_km = -1e-3 * z``.
        """
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
