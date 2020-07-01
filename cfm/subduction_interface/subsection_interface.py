from icp_error.io.array_operations import read_tiff
import geopandas as gpd
import numpy as np

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



