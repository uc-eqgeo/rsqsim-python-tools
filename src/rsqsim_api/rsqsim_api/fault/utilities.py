"""
Geometric utility functions for fault-trace analysis and manipulation.

Provides bearing arithmetic, line operations (strike, dip direction,
reversal, smoothing), and helpers for merging nearly-adjacent
:class:`~shapely.geometry.LineString` segments into single traces.
"""
import os.path

import numpy as np
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd
import difflib


def smallest_difference(value1, value2):
    """
    Return the smallest angular difference between two bearings.

    Accounts for the circular nature of bearings so that, for example,
    the difference between 350° and 10° is 20° rather than 340°.

    Parameters
    ----------
    value1 : float or int
        First bearing in degrees.
    value2 : float or int
        Second bearing in degrees.

    Returns
    -------
    float
        Smallest non-negative angular difference in degrees (0–180).
    """
    abs_diff = abs(value1 - value2)
    if abs_diff > 180:
        smallest_diff = 360 - abs_diff
    else:
        smallest_diff = abs_diff

    return smallest_diff


def normalize_bearing(bearing: float | int):
    """
    Normalise a bearing to the range [0, 360).

    Parameters
    ----------
    bearing : float or int
        Input bearing in degrees (any value).

    Returns
    -------
    float
        Equivalent bearing in the range [0, 360).
    """
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing


def bearing_leq(value: int | float, benchmark: int | float, tolerance: int | float = 0.1):
    """
    Check whether a bearing is anticlockwise of (less than) another bearing.

    Parameters
    ----------
    value : int or float
        The bearing to test, in degrees.
    benchmark : int or float
        The reference bearing, in degrees.
    tolerance : int or float, optional
        Angular tolerance in degrees to account for rounding.
        Defaults to 0.1.

    Returns
    -------
    bool
        ``True`` if ``value`` is strictly anticlockwise of
        ``benchmark`` (i.e. less than, within ``tolerance``).
    """
    smallest_diff = smallest_difference(value, benchmark)
    if smallest_diff > tolerance:
        compare_value = normalize_bearing(value + smallest_diff)
        return abs(compare_value - normalize_bearing(benchmark)) <= tolerance
    else:
        return False


def bearing_geq(value: int | float, benchmark: int | float, tolerance: int | float = 0.1):
    """
    Check whether a bearing is clockwise of (greater than) another bearing.

    Parameters
    ----------
    value : int or float
        The bearing to test, in degrees.
    benchmark : int or float
        The reference bearing, in degrees.
    tolerance : int or float, optional
        Angular tolerance in degrees to account for rounding.
        Defaults to 0.1.

    Returns
    -------
    bool
        ``True`` if ``value`` is strictly clockwise of ``benchmark``
        (i.e. greater than, within ``tolerance``).
    """
    smallest_diff = smallest_difference(value, benchmark)
    if smallest_diff > tolerance:
        compare_value = normalize_bearing(value - smallest_diff)
        return abs(compare_value - normalize_bearing(benchmark)) <= tolerance
    else:
        return False


def reverse_bearing(bearing: int | float):
    """
    Return the bearing 180° opposite to the supplied bearing.

    Parameters
    ----------
    bearing : int or float
        Input bearing in degrees; must be in [0, 360].

    Returns
    -------
    float
        Reversed bearing in the range [0, 360).

    Raises
    ------
    AssertionError
        If ``bearing`` is not a float or int, or is outside [0, 360].
    """
    assert isinstance(bearing, (float, int))
    assert 0. <= bearing <= 360.
    new_bearing = bearing + 180.

    # Ensure strike is between zero and 360 (bearing)
    return normalize_bearing(new_bearing)


def reverse_line(line: LineString):
    """
    Reverse the vertex order of a :class:`~shapely.geometry.LineString`.

    Works with both 2-D and 3-D (has_z) lines.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input line to reverse.

    Returns
    -------
    shapely.geometry.LineString
        New line with vertices in the opposite order.

    Raises
    ------
    AssertionError
        If ``line`` is not a :class:`~shapely.geometry.LineString`.
    """
    assert isinstance(line, LineString)
    if line.has_z:
        x, y, z = np.array(line.coords).T
    else:
        x, y = np.array(line.coords).T
    x_back = x[-1::-1]
    y_back = y[-1::-1]

    if line.has_z:
        z_back = z[-1::-1]
        new_line = LineString([[xi, yi, zi] for xi, yi, zi in zip(x_back, y_back, z_back)])
    else:
        new_line = LineString([[xi, yi] for xi, yi in zip(x_back, y_back)])
    return new_line

def fit_2d_line(x: np.ndarray, y: np.ndarray):
    """
    Fit a straight line to a 2-D point cloud and return the dip angle.

    Fits both ``y = f(x)`` and ``x = g(y)`` and selects whichever has
    the smaller residual, then converts the gradient to a dip angle.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of the points.
    y : numpy.ndarray
        y-coordinates of the points.

    Returns
    -------
    float
        Dip angle in degrees measured from the horizontal.

    Raises
    ------
    AssertionError
        If either ``x`` or ``y`` is not a :class:`numpy.ndarray`.
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)

    px = np.polyfit(x, y, 1, full=True)
    gradient_x = px[0][0]

    if len(px[1]):
        res_x = px[1][0]
    else:
        res_x = 0

    py = np.polyfit(y, x, 1, full=True)
    gradient_y = py[0][0]
    if len(py[1]):
        res_y = py[1][0]
    else:
        res_y = 0

    if res_x <= res_y:
        dip_angle = np.degrees(np.arctan(gradient_x))
    else:
        dip_angle = np.degrees(np.arctan(1./gradient_y))

    return dip_angle


def calculate_dip_direction(line: LineString):
    """
    Calculate the dip direction of a fault-trace LineString in NZTM.

    Computes the strike of the line (best-fit gradient) and adds 90° to
    obtain the dip direction.  The sign is chosen so that the dip
    direction is consistent with the majority of vertices being to the
    right of the strike vector.

    Parameters
    ----------
    line : shapely.geometry.LineString or MultiLineString
        Fault-surface trace in NZTM (EPSG:2193) coordinates.
        A :class:`~shapely.geometry.MultiLineString` is first merged
        into a single line.

    Returns
    -------
    float
        Dip direction in degrees, range [0, 360).

    Raises
    ------
    AssertionError
        If ``line`` is not a LineString or MultiLineString.
    """
    assert isinstance(line, (LineString, MultiLineString))
    if isinstance(line, MultiLineString):
        line = merge_multiple_nearly_adjacent_segments(list(line.geoms))
    # Get coordinates
    x, y = line.xy
    x, y = np.array(x), np.array(y)
    # Calculate gradient of line in 2D
    px = np.polyfit(x, y, 1, full=True)
    gradient_x = px[0][0]

    if len(px[1]):
        res_x = px[1][0]
    else:
        res_x = 0

    py = np.polyfit(y, x, 1, full=True)
    gradient_y = py[0][0]
    if len(py[1]):
        res_y = py[1][0]
    else:
        res_y = 0

    if res_x <= res_y:
        # Gradient to bearing
        bearing = 180 - np.degrees(np.arctan2(gradient_x, 1))
    else:
        bearing = 180 - np.degrees(np.arctan2(1/gradient_y, 1))

    strike = normalize_bearing(bearing - 90.)
    strike_vector = np.array([np.sin(np.radians(strike)), np.cos(np.radians(strike))])

    # Determine whether line object fits strike convention
    relative_x = x - x[0]
    relative_y = y - y[0]

    distances = np.matmul(np.vstack((relative_x, relative_y)).T, strike_vector)
    num_pos = np.count_nonzero(distances > 0)
    num_neg = np.count_nonzero(distances <= 0)

    if num_neg > num_pos:
        bearing += 180.

    dip_direction = bearing
    # Ensure strike is between zero and 360 (bearing)
    while dip_direction < 0:
        dip_direction += 360.

    while dip_direction >= 360.:
        dip_direction -= 360.

    return dip_direction

def calculate_strike(line: LineString, lt180: bool = False):
    """
    Calculate the strike of a fault-trace LineString in NZTM.

    Fits a best-fit gradient to the trace coordinates and converts it
    to an azimuthal strike.  When ``lt180`` is ``True`` the result is
    reduced to the half-circle convention [0, 180).

    Parameters
    ----------
    line : shapely.geometry.LineString or MultiLineString
        Fault-surface trace in NZTM (EPSG:2193) coordinates.
        A :class:`~shapely.geometry.MultiLineString` is first merged
        into a single line.
    lt180 : bool, optional
        If ``True``, return a strike in [0, 180) instead of [0, 360).
        Defaults to ``False``.

    Returns
    -------
    float
        Strike in degrees.

    Raises
    ------
    AssertionError
        If ``line`` is not a LineString or MultiLineString.
    """
    assert isinstance(line, (LineString, MultiLineString))
    if isinstance(line, MultiLineString):
        line = merge_multiple_nearly_adjacent_segments(list(line.geoms))
    # Get coordinates
    x, y = line.xy
    x, y = np.array(x), np.array(y)
    if (y==y[0]).all():
        bearing = 180.
    elif (x==x[0]).all():
        bearing = 90.

    else:
        # Calculate gradient of line in 2D
        px = np.polyfit(x, y, 1, full=True)
        gradient_x = px[0][0]

        if len(px[1]):
            res_x = px[1][0]
        else:
            res_x = 0


        py = np.polyfit(y, x, 1, full=True)
        gradient_y = py[0][0]
        if len(py[1]):
            res_y = py[1][0]
        else:
            res_y = 0

        if res_x <= res_y:
            # Gradient to bearing
            bearing = 180 - np.degrees(np.arctan2(gradient_x, 1))
        else:
            bearing = 180 - np.degrees(np.arctan2(1/gradient_y, 1))

    strike = normalize_bearing(bearing - 90.)

    # Ensure strike is between zero and 360 (bearing)
    while strike < 0:
        strike += 360.

    while strike >= 360.:
        strike -= 360.

    if lt180:
        while strike >= 180.:
            strike -= 180.

        while strike < 0:
            strike += 180.

    return strike


def optimize_point_spacing(line: LineString, spacing: float):
    """
    Re-sample a LineString to approximately uniform point spacing.

    Interpolates ``num_points`` centre-points along the line, where
    ``num_points`` is chosen to match ``spacing`` as closely as
    possible.

    Parameters
    ----------
    line : shapely.geometry.LineString or MultiLineString
        Input line.  A :class:`~shapely.geometry.MultiLineString` is
        first merged into a single line.
    spacing : float
        Target point spacing in the same units as the line coordinates
        (metres for NZTM).

    Returns
    -------
    centre_points : list of shapely.geometry.Point
        Resampled centre points along the line.
    new_width : float
        Actual spacing used (line length / number of points).

    Raises
    ------
    AssertionError
        If ``line`` is not a LineString or MultiLineString, or if
        ``spacing`` is not a positive float.
    """
    assert isinstance(line, (LineString, MultiLineString))
    if isinstance(line, MultiLineString):
        line = merge_multiple_nearly_adjacent_segments(list(line.geoms))
    assert isinstance(spacing, float)
    assert spacing > 0.
    line_length = line.length
    num_points = int(np.round(line_length / spacing, 0))
    if num_points ==0:
        num_points=1
    new_width = line_length / num_points
    centre_points = [line.interpolate((i + 0.5) * new_width) for i in range(num_points)]
    return centre_points, new_width



import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import linemerge



def chaikins_corner_cutting(coords, refinements=5):
    """
    Smooth a polyline using Chaikin's corner-cutting algorithm.

    Each refinement pass replaces every edge with two new points at
    the 1/4 and 3/4 positions, progressively rounding corners.

    Parameters
    ----------
    coords : array-like of shape (n, 2) or (n, 3)
        Coordinate array of the polyline vertices.
    refinements : int, optional
        Number of corner-cutting passes to apply.  Defaults to 5.

    Returns
    -------
    numpy.ndarray
        Smoothed coordinate array with approximately
        ``n * 2 ** refinements`` vertices.
    """
    coords = np.array(coords)

    for _ in range(refinements):
        l = coords.repeat(2, axis=0)
        R = np.empty_like(l)
        R[0] = l[0]
        R[2::2] = l[1:-1:2]
        R[1:-1:2] = l[2::2]
        R[-1] = l[-1]
        coords = l * 0.75 + R * 0.25

    return coords

def smooth_trace(trace: LineString, n_refinements: int = 5):
    """
    Smooth a fault-trace LineString using Chaikin's corner-cutting algorithm.

    Parameters
    ----------
    trace : shapely.geometry.LineString
        Input trace to smooth.
    n_refinements : int, optional
        Number of corner-cutting passes.  Defaults to 5.

    Returns
    -------
    shapely.geometry.LineString
        Smoothed line with approximately
        ``len(trace.coords) * 2 ** n_refinements`` vertices.

    Raises
    ------
    AssertionError
        If ``trace`` is not a :class:`~shapely.geometry.LineString`.
    """
    assert isinstance(trace, LineString)
    coords = np.array(trace.coords)
    return LineString(chaikins_corner_cutting(coords, refinements=n_refinements))


def straighten(line: LineString, strike: float, damping: float):
    """
    Straighten a 3-D line by damping its across-strike deviations.

    Projects each vertex onto the along-strike and across-strike
    directions, then reconstructs new positions with the across-strike
    component multiplied by ``damping``.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input 3-D line (z coordinate included in
        :attr:`~shapely.geometry.LineString.coords`).
    strike : float
        Strike azimuth in degrees used to define the along/across
        direction.
    damping : float
        Fractional weight applied to the across-strike displacement
        (0 = fully straight, 1 = unchanged).

    Returns
    -------
    shapely.geometry.LineString
        New line with reduced across-strike curvature.
    """
    strike_vector = np.array([np.sin(np.radians(strike)), np.cos(np.radians(strike)), 0.])
    across_strike = np.array([np.sin(np.radians(strike + 90.)), np.cos(np.radians(strike + 90.)), 0.])
    line_array = np.array(line.coords)
    centroid = np.array(line.centroid)

    along_dists = np.dot(line_array - centroid, strike_vector)
    across_dists = np.dot(line_array - centroid, across_strike)

    new_locations = centroid + along_dists + damping * across_dists

    return LineString(new_locations)


def align_two_nearly_adjacent_segments(segment_list: list[LineString], tolerance: float = 200.):
    """
    Snap the closest endpoints of two nearly-adjacent LineStrings to their midpoint.

    Identifies which endpoints of the two segments are closest to each
    other and replaces both with the midpoint, making the segments
    exactly adjacent.

    Parameters
    ----------
    segment_list : list of shapely.geometry.LineString
        Exactly two line segments.
    tolerance : float, optional
        Maximum allowable distance (m) between the nearest endpoints
        for the operation to be valid.  Defaults to 200.

    Returns
    -------
    tuple of shapely.geometry.LineString
        Two new segments with snapped endpoints.

    Raises
    ------
    AssertionError
        If ``segment_list`` does not contain exactly 2 segments, or
        if the distance between them exceeds ``tolerance``.
    """
    assert len(segment_list) == 2
    line1, line2 = segment_list
    assert line1.distance(line2) <= tolerance

    l1e1 = Point(line1.coords[0])
    l1e2 = Point(line1.coords[-1])
    p1 = l1e1 if l1e1.distance(line2) <= l1e2.distance(line2) else l1e2

    l2e1 = Point(line2.coords[0])
    l2e2 = Point(line2.coords[-1])

    p2 = l2e1 if l2e1.distance(line1) <= l2e2.distance(line1) else l2e2

    mid_point = Point(0.5 * (np.array(p1.coords) + np.array(p2.coords)).flatten())

    if l1e1 == p1:
        new_line1 = np.vstack([np.array(mid_point.coords), np.array(line1.coords)[1:]])
    else:
        new_line1 = np.vstack([np.array(line1.coords)[:-1], np.array(mid_point.coords)])

    if l2e1 == p2:
        new_line2 = np.vstack([np.array(mid_point.coords), np.array(line2.coords)[1:]])
    else:
        new_line2 = np.vstack([np.array(line2.coords)[:-1], np.array(mid_point.coords)])

    return LineString(new_line1), LineString(new_line2)

def merge_two_nearly_adjacent_segments(segment_list: list[LineString], tolerance: float = 200.):
    """
    Merge two nearly-adjacent LineStrings into one by snapping their endpoints.

    Calls :func:`align_two_nearly_adjacent_segments` to snap the
    closest endpoints to their midpoint, then merges the result into a
    single :class:`~shapely.geometry.LineString` with
    :func:`~shapely.ops.linemerge`.

    Parameters
    ----------
    segment_list : list of shapely.geometry.LineString
        Exactly two line segments.
    tolerance : float, optional
        Maximum allowable gap (m) between segments.  Defaults to 200.

    Returns
    -------
    shapely.geometry.LineString
        Merged single line.
    """
    new_line1, new_line2 = align_two_nearly_adjacent_segments(segment_list, tolerance)
    return linemerge([new_line1, new_line2])


def merge_multiple_nearly_adjacent_segments(segment_list: list[LineString], tolerance: float = 200.):
    """
    Iteratively merge a list of nearly-adjacent LineStrings into one.

    Repeatedly merges the first two segments using
    :func:`merge_two_nearly_adjacent_segments` until a single line
    remains.

    Parameters
    ----------
    segment_list : list of shapely.geometry.LineString
        Two or more line segments to merge, ordered along the trace.
    tolerance : float, optional
        Maximum allowable gap (m) between consecutive segments.
        Defaults to 200.

    Returns
    -------
    shapely.geometry.LineString
        Single merged line.

    Raises
    ------
    AssertionError
        If ``segment_list`` contains fewer than 2 segments.
    """
    assert len(segment_list) >= 2
    if len(segment_list) == 2:
        return merge_two_nearly_adjacent_segments(segment_list, tolerance)
    else:
        while len(segment_list) > 2:
            segment_list = [merge_two_nearly_adjacent_segments(segment_list[:2], tolerance)] + segment_list[2:]
        return merge_two_nearly_adjacent_segments(segment_list, tolerance)
