from icp_error.io.array_operations import read_tiff
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString


def nan_helper(a):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, b= nan_helper(a)
        >>> b[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(a), lambda b: b.nonzero()[0]


x, y, z = read_tiff("hik_res_0.01_lt50_nztm.tif")
z *= 1000

x_grid, y_grid = np.meshgrid(x, y)

overall_trace = gpd.GeoDataFrame.from_file("overall_trace.shp")
overall_line = overall_trace.geometry[0]

corner = np.array(overall_line.coords[0])
overall_vec = np.array(overall_line.coords[1]) - corner

overall_strike = 90 - np.degrees(np.arctan2(overall_vec[1], overall_vec[0]))
along_overall = overall_vec / np.linalg.norm(overall_vec)

across_vec = np.matmul(np.array([[0, -1], [1, 0]]), along_overall)

along_dists = (x_grid - corner[0]) * along_overall[0] + (y_grid - corner[1]) * along_overall[1]
across_dists = (x_grid - corner[0]) * across_vec[0] + (y_grid - corner[1]) * across_vec[1]

profile_half_width = 1000
profile_spacing = 7000

# Find start location
start_along = min(along_dists[~np.isnan(z)])
end_along = max(along_dists[~np.isnan(z)])

along_spaced = np.arange(start_along + profile_spacing/2, end_along, profile_spacing)

for along in along_spaced:
    row_end = corner + along * along_overall

    along_min = along - profile_half_width
    along_max = along + profile_half_width

    in_swath = np.logical_and(along_dists >= along_min, along_dists <= along_max)
    swath_across = across_dists[in_swath]
    swath_z = z[in_swath]

    across_no_nans = swath_across[~np.isnan(swath_z)]

    start_across = min(across_no_nans)
    end_across = max(across_no_nans)

    initial_spacing = np.arange(start_across, end_across, profile_half_width)

    across_vs_z = np.vstack((across_no_nans, swath_z[~np.isnan(swath_z)])).T
    sorted_coords = across_vs_z[across_vs_z[:, 0].argsort()]

    interp_z = np.interp(initial_spacing, sorted_coords[:, 0], sorted_coords[:, 1])

    interp_line = LineString(np.vstack((initial_spacing, interp_z)).T)

    interpolation_distances = np.arange(profile_spacing/2, interp_line.length, profile_spacing)
    interpolated_points = [interp_line.interpolate(distance) for distance in interpolation_distances]

    interpolated_x = np.array([point.x for point in interpolated_points])
    interpolated_z_values = np.array([point.y for point in interpolated_points])

    point_xys = np.array([row_end + across_i * across_vec for across_i in interpolated_x])
    point_xyz = np.vstack((point_xys.T, interpolated_z_values)).T







