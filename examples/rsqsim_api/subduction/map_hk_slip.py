import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import os
import pandas as pd

from pyproj import Transformer
from rsqsim_api.fault.segment import RsqSimSegment

transformer = Transformer.from_crs(4326, 2193, always_xy=True)
trans_inv = Transformer.from_crs(2193, 4326, always_xy=True)

data_dir = "../../../data/subduction"

# Read shapefile: line that gives overall strike of subduction zone
# Included so that easy to fiddle with in GIS
overall_trace = gpd.GeoDataFrame.from_file(os.path.join(data_dir, "overall_trace.shp"))
overall_line = overall_trace.geometry[0]
# Get SE corner of line
corner = np.array(overall_line.coords[0])
# In case (for example) geopandas is not installed
# corner = np.array([1627783.3117604, 4942542.56366084])
# Direction of trace in vector form
overall_vec = np.array(overall_line.coords[1]) - corner
# Turn into degrees for inspection of values
overall_strike = 90 - np.degrees(np.arctan2(overall_vec[1], overall_vec[0]))
# Unit vector along strike
along_overall = overall_vec / np.linalg.norm(overall_vec)
# Rotate to give unit vector perpendicular to strike
across_vec = np.matmul(np.array([[0, -1], [1, 0]]), along_overall)

output_dir = os.getcwd()



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





def point_dist(point: Point):
    return np.dot(along_overall, np.array(transformer.transform(point.x, point.y)))

def point_dist_nztm(point: Point):
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


stl_file = os.path.join(data_dir, "hik_kerm_10km_trimmed.stl")
fault = RsqSimSegment.from_stl(stl_file)

slip_deficit_file = os.path.join(data_dir, "trench_creep_hik_slipdeficit.txt")
slip_deficit_df = pd.read_csv(slip_deficit_file, delim_whitespace=True)
slip_deficit_nztm_x, slip_deficit_nztm_y = transformer.transform(slip_deficit_df["#long"].to_list(),
                                                                 slip_deficit_df["lat"].to_list())
slip_deficit_nztm_array = np.vstack((slip_deficit_nztm_x, slip_deficit_nztm_y, slip_deficit_df.depth.to_numpy() * 1000.,
                                     slip_deficit_df["slip_deficit_mm/yr"].to_numpy(),
                                     slip_deficit_df["uncertainty_mm/yr"].to_numpy())).T

data_half_width = 5000.

for patch in fault.patch_outlines:
    centroid = patch.centre
    centre_dist = point_dist_nztm(Point(centroid[0], centroid[1]))
    # Find slip deficit
    sd_difference_vectors = slip_deficit_nztm_array[:, :3] - centroid
    sd_all_values = slip_deficit_nztm_array[:, 3]
    sd_distances = np.linalg.norm(sd_difference_vectors, axis=1)
    nearby_sd = slip_deficit_nztm_array[sd_distances < data_half_width]
    if nearby_sd.size:
        gaussian_sigma = data_half_width / 3.
        sd_weights = np.exp(-1 * sd_distances[sd_distances < data_half_width] / (2 * gaussian_sigma))
        sd_values = sd_all_values[sd_distances < data_half_width]
        sd_average = np.average(sd_values, weights=sd_weights)
        # if centre_dist > east_cape_dist:
        #     sd_average = kermadec_slip_rate(float(centre_dist), modelled_value=sd_average)
    elif centre_dist > start_0_2_dist:
        # print(centre_wgs)
        sd_average = kermadec_slip_rate(float(centre_dist))
    else:
        sd_average = 0.

    patch.dip_slip = sd_average





