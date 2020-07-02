from icp_error.io.array_operations import read_tiff
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm


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
profile_spacing = 25000

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

all_tile_ls = []

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

    all_tile_ls.append(np.array(poly_ls))

all_polygons = [Polygon(array_i) for array_i in all_tile_ls]

outlines = gpd.GeoSeries(all_polygons, crs="epsg:2193")
outlines.to_file("tile_outlines.shp")

all_points = [Point(row) for row in all_points_array]
centres = gpd.GeoSeries(all_points, crs="epsg:2193")
centres.to_file("tile_centres.shp")
all_points_z = np.array([point.z for point in all_points])


patch_colour = "b"
patch_alpha = 0.5
line_colour = "k"
line_width = 0.2
vertical_exaggeration = 10

collection_list = []
for corners in all_tile_ls:
    xc, yc, zc = corners.T.tolist()
    collection_list.append(list(zip(xc, yc, zc)))

patch_collection = Poly3DCollection(collection_list, alpha=patch_alpha, facecolors=patch_colour)
line_collection = Line3DCollection(collection_list, linewidths=line_width, colors=line_colour)

colormap = cm.ScalarMappable(cmap=cm.magma)
colormap.set_array(np.array([min(all_points_z), max(all_points_z)]))
colormap.set_clim(vmin=min(all_points_z), vmax=max(all_points_z))

patch_collection.set_facecolor(colormap.to_rgba(all_points_z, alpha=patch_alpha))

plt.close("all")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


x1, y1 = min(x), min(y)
x_range = max(x) - x1
y_range = max(y) - y1

plot_width = max([x_range, y_range])

x2 = x1 + plot_width
y2 = y1 + plot_width

ax.add_collection3d(patch_collection)
ax.add_collection3d(line_collection)
ax.set_ylim((y1, y2))
ax.set_xlim((x1, x2))
ax.set_zlim((-(1/vertical_exaggeration) * plot_width, 10))

fig.show()
