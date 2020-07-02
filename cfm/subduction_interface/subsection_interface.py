from icp_error.io.array_operations import read_tiff
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point

def fit_plane_svd(point_cloud):
    G = point_cloud.sum(axis=0) / point_cloud.shape[0]

    # run SVD
    u, s, vh = np.linalg.svd(point_cloud - G)

    # unitary normal vector
    u_norm = vh[2, :]
    return u_norm


x, y, z = read_tiff("hik_res_0.01_lt50_nztm.tif")
z *= 1000

x_grid, y_grid = np.meshgrid(x, y)

all_xyz_with_nans = np.vstack((x_grid.flatten(), y_grid.flatten(), z.flatten())).T
all_xyz = all_xyz_with_nans[~np.isnan(all_xyz_with_nans).any(axis=1)]



overall_trace = gpd.GeoDataFrame.from_file("overall_trace.shp")
overall_line = overall_trace.geometry[0]

corner = np.array(overall_line.coords[0])
overall_vec = np.array(overall_line.coords[1]) - corner

overall_strike = 90 - np.degrees(np.arctan2(overall_vec[1], overall_vec[0]))
along_overall = overall_vec / np.linalg.norm(overall_vec)

across_vec = np.matmul(np.array([[0, -1], [1, 0]]), along_overall)

along_dists = (x_grid - corner[0]) * along_overall[0] + (y_grid - corner[1]) * along_overall[1]
across_dists = (x_grid - corner[0]) * across_vec[0] + (y_grid - corner[1]) * across_vec[1]

profile_half_width = 2000
profile_spacing = 7000

# Find start location
start_along = min(along_dists[~np.isnan(z)])
end_along = max(along_dists[~np.isnan(z)])

along_spaced = np.arange(start_along + profile_spacing/2, end_along, profile_spacing)

all_points_ls = []

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

    all_points_ls.append(point_xyz)

all_points_array = np.vstack(all_points_ls)

search_radius = 1e4

all_polygons = []

for centre_point in all_points_array:
    difference_vectors = all_xyz - centre_point
    distances = np.linalg.norm(difference_vectors, axis=1)
    small_cloud = all_xyz[distances < search_radius]

    u_norm_i = fit_plane_svd(small_cloud)

    # normal_i = np.array(model_i[:-1])
    #
    normal_i = u_norm_i
    if normal_i[-1] < 0:
        normal_i *= -1

    strike_vector = np.cross(normal_i, np.array([0, 0, -1]))
    strike_vector[-1] = 0
    strike_vector /= np.linalg.norm(strike_vector)

    down_dip_vector = np.cross(normal_i, strike_vector)
    if down_dip_vector[-1] > 0:
        down_dip_vector *= -1

    dip = np.degrees(np.arctan(-1 * down_dip_vector[-1] / np.linalg.norm(down_dip_vector[:-1])))

    poly_ls = []
    for i, j in zip([1, 1, -1, -1], [1, -1, -1, 1]):
        corner_i = centre_point + (i * strike_vector + j * down_dip_vector) * profile_spacing / 2
        poly_ls.append(corner_i)

    all_polygons.append(Polygon(poly_ls))

outlines = gpd.GeoSeries(all_polygons, crs="epsg:2193")
outlines.to_file("tile_outlines.shp")

all_points = [Point(row) for row in all_points_array]
centres = gpd.GeoSeries(all_points, crs="epsg:2193")
centres.to_file("tile_centres.shp")

