"""
Hikurangi subduction zone slip-rate and coupling utilities.

Provides helper functions for computing along-trench distances,
coupling coefficients, and convergence rates for the Hikurangi margin,
as well as a general best-fit-plane routine.
"""
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
    Fit a best-fit plane to a set of 3-D points using SVD.

    The plane is constrained to pass through the centroid of the
    point cloud.  The normal is extracted from the last column of the
    left singular matrix of the de-meaned covariance and is oriented
    so that its z-component is positive.

    Parameters
    ----------
    points : numpy.ndarray of shape (n, 3)
        Array of 3-D point coordinates.
    eps : float, optional
        Threshold below which normal-vector components are set to
        exactly zero.  Defaults to 1e-5.

    Returns
    -------
    plane_normal : numpy.ndarray of shape (3,)
        Unit normal vector to the best-fit plane (A, B, C).
    plane_origin : numpy.ndarray of shape (3,)
        Centroid of the input points, lying on the fitted plane.
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
    """
    Project a WGS84 point onto an NZTM direction vector.

    Transforms the point to NZTM (EPSG:2193) and computes its dot
    product with the supplied unit direction vector.

    Parameters
    ----------
    point : shapely.geometry.Point
        Point in WGS84 (longitude, latitude) coordinates.
    along_overall : numpy.ndarray of shape (2,)
        2-D unit direction vector in NZTM coordinates.

    Returns
    -------
    float
        Scalar projection of the transformed point onto
        ``along_overall``.
    """
    return np.dot(along_overall, np.array(transformer.transform(point.x, point.y)))

def point_dist_nztm(point: Point, along_overall: np.ndarray):
    """
    Project an NZTM point onto a direction vector.

    Parameters
    ----------
    point : shapely.geometry.Point
        Point already in NZTM (EPSG:2193) coordinates.
    along_overall : numpy.ndarray of shape (2,)
        2-D unit direction vector.

    Returns
    -------
    float
        Scalar projection of the point onto ``along_overall``.
    """
    return float(np.dot(along_overall, np.array([point.x, point.y])))

east_cape_dist = point_dist(east_cape)
start_0_2_dist = point_dist(start_0_2)
end_0_2_dist = point_dist(end_0_2)
start_0_5_dist = point_dist(start_0_5)
convergence_start_dist = point_dist(convergence_start)
convergence_end_dist = point_dist(convergence_end)

def coupling(dist: float):
    """
    Return the Hikurangi plate-interface coupling coefficient at a given along-trench distance.

    Implements a piecewise model: 0.2 south of ``start_0_2``, 0.5
    north of ``start_0_5``, and a linear gradient in between.

    Parameters
    ----------
    dist : float
        Along-trench distance (m in NZTM projection) from east cape.

    Returns
    -------
    float
        Coupling coefficient (dimensionless, 0–1).

    Raises
    ------
    AssertionError
        If ``dist`` is south of east cape (i.e. less than
        ``east_cape_dist``).
    """
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
    Return the Hikurangi convergence rate at a given along-trench distance.

    Linear interpolation between 49 mm/yr at the southern end
    (``convergence_start``) and 85 mm/yr at the northern end
    (``convergence_end``).

    Parameters
    ----------
    dist : float
        Along-trench distance (m in NZTM projection).

    Returns
    -------
    float
        Convergence rate in mm/yr.
    """
    south_conv = 49.
    north_conv = 85.

    return south_conv + (north_conv - south_conv) * (dist - convergence_start_dist) / (convergence_end_dist -
                                                                                       convergence_start_dist)


def convergence_dist(dist):
    pass

def kermadec_slip_rate(dist: float, modelled_value: float = 0.):
    """
    Compute the Kermadec subduction zone slip rate at a given along-trench distance.

    Blends the modelled slip rate with the convergence-rate-based
    estimate using a linear fractional interpolation from east cape to
    the 0.2-coupling start point.  If ``modelled_value`` is zero,
    returns the pure convergence × coupling estimate.

    Parameters
    ----------
    dist : float
        Along-trench distance (m in NZTM projection).
    modelled_value : float, optional
        Modelled slip rate (mm/yr) from the fault model.  If 0
        (default), only coupling × convergence is used.

    Returns
    -------
    float
        Slip rate in mm/yr.
    """
    if modelled_value > 0.:
        frac = (dist - east_cape_dist) / (start_0_2_dist - east_cape_dist)
        print(frac)
        return modelled_value * (1 - frac) + convergence(dist) * coupling(dist) * frac
    else:
        return convergence(dist) * coupling(dist)
