import os.path

import numpy as np
from typing import Union, List, Tuple, Dict
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd
import difflib


def smallest_difference(value1, value2):
    """
    Finds smallest angle between two bearings
    :param value1:
    :param value2:
    :return:
    """
    abs_diff = abs(value1 - value2)
    if abs_diff > 180:
        smallest_diff = 360 - abs_diff
    else:
        smallest_diff = abs_diff

    return smallest_diff


def normalize_bearing(bearing: Union[float, int]):
    """
    change a bearing (in degrees) so that it is an azimuth between 0 and 360.
    :param bearing:
    :return:
    """
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing


def bearing_leq(value: Union[int, float], benchmark: Union[int, float], tolerance: Union[int, float] = 0.1):
    """
    Check whether a bearing (value) is anticlockwise of another bearing (benchmark)
    :param value:
    :param benchmark:
    :param tolerance: to account for rounding errors etc
    :return:
    """
    smallest_diff = smallest_difference(value, benchmark)
    if smallest_diff > tolerance:
        compare_value = normalize_bearing(value + smallest_diff)
        return abs(compare_value - normalize_bearing(benchmark)) <= tolerance
    else:
        return False


def bearing_geq(value: Union[int, float], benchmark: Union[int, float], tolerance: Union[int, float] = 0.1):
    """
    Check whether a bearing (value) is clockwise of another bearing (benchmark)
    :param value:
    :param benchmark:
    :param tolerance: to account for rounding errors etc
    :return:
    """
    smallest_diff = smallest_difference(value, benchmark)
    if smallest_diff > tolerance:
        compare_value = normalize_bearing(value - smallest_diff)
        return abs(compare_value - normalize_bearing(benchmark)) <= tolerance
    else:
        return False


def reverse_bearing(bearing: Union[int, float]):
    """
    180 degrees from supplied bearing
    :param bearing:
    :return:
    """
    assert isinstance(bearing, (float, int))
    assert 0. <= bearing <= 360.
    new_bearing = bearing + 180.

    # Ensure strike is between zero and 360 (bearing)
    return normalize_bearing(new_bearing)


def reverse_line(line: LineString):
    """
    Change the order that points in a LineString object are presented.
    Updated to work with 3d lines (has_z), September 2021
    Important for OpenSHA, I think
    :param line:
    :return:
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
    Fit a 2D line to a set of points
    :param x:
    :param y:
    :return:
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
    Calculate the strike of a shapely linestring object with coordinates in NZTM,
    then adds 90 to get dip direction. Dip direction is always 90 clockwise from strike of line.
    :param line: Linestring object
    :return:
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
    Calculate the strike of a shapely linestring object with coordinates in NZTM
    :param line: Linestring object
    :return:
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
    Optimize point spacing of a linestring object. Might be less good than geopandas segmentize.
    :param line:
    :param spacing:
    :return:
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
from typing import List
from shapely.geometry import LineString, Point
from shapely.ops import linemerge



def chaikins_corner_cutting(coords, refinements=5):
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
    assert isinstance(trace, LineString)
    coords = np.array(trace.coords)
    return LineString(chaikins_corner_cutting(coords, refinements=n_refinements))


def straighten(line: LineString, strike: float, damping: float):
    strike_vector = np.array([np.sin(np.radians(strike)), np.cos(np.radians(strike)), 0.])
    across_strike = np.array([np.sin(np.radians(strike + 90.)), np.cos(np.radians(strike + 90.)), 0.])
    line_array = np.array(line.coords)
    centroid = np.array(line.centroid)

    along_dists = np.dot(line_array - centroid, strike_vector)
    across_dists = np.dot(line_array - centroid, across_strike)

    new_locations = centroid + along_dists + damping * across_dists

    return LineString(new_locations)


def align_two_nearly_adjacent_segments(segment_list: List[LineString], tolerance: float = 200.):
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

def merge_two_nearly_adjacent_segments(segment_list: List[LineString], tolerance: float = 200.):
    new_line1, new_line2 = align_two_nearly_adjacent_segments(segment_list, tolerance)
    return linemerge([new_line1, new_line2])


def merge_multiple_nearly_adjacent_segments(segment_list: List[LineString], tolerance: float = 200.):
    assert len(segment_list) >= 2
    if len(segment_list) == 2:
        return merge_two_nearly_adjacent_segments(segment_list, tolerance)
    else:
        while len(segment_list) > 2:
            segment_list = [merge_two_nearly_adjacent_segments(segment_list[:2], tolerance)] + segment_list[2:]
        return merge_two_nearly_adjacent_segments(segment_list, tolerance)










