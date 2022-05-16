import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import os
import pandas as pd

from pyproj import Transformer

transformer = Transformer.from_crs(4326, 2193, always_xy=True)
trans_inv = Transformer.from_crs(2193, 4326, always_xy=True)


def fit_plane_to_points(points: np.ndarray, eps: float=1.0e-5):
    """
    Find best-fit plane through a set of points, after first insuring the plane goes through
    the mean (centroid) of all the points in the array. This is probably better than my
    initial method, since the SVD is only over a 3x3 array (rather than the num_pointsxnum_points
    array).
    Returned values are:
        plane_normal:  Normal vector to plane (A, B, C)
        plane_origin:  Point on plane that may be considered as the plane origin
    """
    # Compute plane origin and subract it from the points array.
    plane_origin = np.mean(points, axis=0)
    x = points - plane_origin

    # Dot product to yield a 3x3 array.
    moment = np.dot(x.T, x)

    # Extract single values from SVD computation to get normal.
    plane_normal = np.linalg.svd(moment)[0][:,-1]
    small = np.where(np.abs(plane_normal) < eps)
    plane_normal[small] = 0.0
    plane_normal /= np.linalg.norm(plane_normal)
    if (plane_normal[-1] < 0.0):
        plane_normal *= -1.0

    return plane_normal, plane_origin


# Locations of points where slip rate changes
east_cape = Point(178.9916, -39.1775)
start_0_2 = Point(180.0, -37.00)
end_0_2 = Point(-177.3995, -32.5061)
start_0_5 = Point(-176.673, -31.016)
convergence_start = Point(179.098, -39.014)
convergence_end = Point(-174.162, -27.508)





def point_dist(point: Point, along_overall: np.ndarray):
    return np.dot(along_overall, np.array(transformer.transform(point.x, point.y)))

def point_dist_nztm(point: Point, along_overall: np.ndarray):
    return float(np.dot(along_overall, np.array([point.x, point.y])))

east_cape_dist = point_dist(east_cape)
start_0_2_dist = point_dist(start_0_2)
end_0_2_dist = point_dist(end_0_2)
start_0_5_dist = point_dist(start_0_5)
convergence_start_dist = point_dist(convergence_start)
convergence_end_dist = point_dist(convergence_end)

def coupling(dist: float):
    assert dist >= east_cape_dist
    if dist < start_0_2_dist:
        return 0.2  # * (dist - east_cape_dist) / (start_0_2_dist - east_cape_dist)
    elif dist < end_0_2_dist:
        return 0.2

    elif dist > start_0_5_dist:
        return 0.5
    else:
        # Linear gradient in the middle between two uniform areas
        return 0.2 + (0.5 - 0.2) * (dist - end_0_2_dist) / (start_0_5_dist - end_0_2_dist)

def convergence(dist: float):
    """
    Linear between 49 mm/yr at -39 to 85 mm/yr at -27.5
    """
    south_conv = 49.
    north_conv = 85.

    return south_conv + (north_conv - south_conv) * (dist - convergence_start_dist) / (convergence_end_dist -
                                                                                       convergence_start_dist)


def convergence_dist(dist):
    pass

def kermadec_slip_rate(dist: float, modelled_value: float = 0.):
    if modelled_value > 0.:
        frac = (dist - east_cape_dist) / (start_0_2_dist - east_cape_dist)
        print(frac)
        return modelled_value * (1 - frac) + convergence(dist) * coupling(dist) * frac
    else:
        return convergence(dist) * coupling(dist)