import os
import numpy as np
from rsqsim_api.containers.fault import RsqSimSegment
import rsqsim_api.io.write_utils as rwu

import meshio

from eq_fault_geom import faultmeshio

import pdb
pdb.set_trace()

# Locate files relative to script
run_dir = os.path.dirname(__file__)
jordan_dxf = os.path.join(run_dir, "../../../data/cfm_dxf/Jordan.dxf")
wellington_tsurf = os.path.join(run_dir, "../../../data/cfm_tsurf/Wellington_Hutt_Valley_1.ts")
white_island_tsurf = os.path.join(run_dir, "../../../data/cfm_tsurf/White_Island_North_2_.ts")

# Read in Jordan fault
fault1 = RsqSimSegment.from_dxf(jordan_dxf)

# Get points and compute normal.
points1 = np.array(fault1.vertices)
(plane_normal1, plane_origin1) = rwu.fit_plane_to_points(points1)

# Get rotation matrix.
rotation_matrix1 = rwu.get_fault_rotation_matrix(plane_normal1)

# Convert to local coordinates.
edges1 = fault1.boundary
(points_local1, edges_local1, fault_is_plane1) = rwu.fault_global_to_local(points1, edges1, rotation_matrix1, plane_origin1)

# Get edges of boundary and create grid in local coordinates.
resolution = 3000.0
num_search_tris = 10
quad_edges1 = rwu.get_quad_mesh_edges(edges_local1)
(mesh_points_local1,
 num_horiz_points, num_vert_points) = rwu.create_local_grid(points_local1, quad_edges1, fault1.triangles,
                                                            fault_is_plane1, resolution=resolution, num_search_tris=num_search_tris)

# Convert to global coordinates and output both original mesh and quad mesh.
(mesh_points_global1, edges_global1) = rwu.fault_local_to_global(mesh_points_local1, edges_local1,
                                                                 rotation_matrix1, plane_origin1)
triangles1 = [("triangle", fault1.triangles)]
quad_cells1 = rwu.create_cells_from_dims(num_horiz_points, num_vert_points)
quads1 = [("quad", quad_cells1)]
mesh_tri1 = meshio.Mesh(points1, triangles1)
out_tri1 = "Jordan_tri_global.vtk"
meshio.write(out_tri1, mesh_tri1, file_format="vtk", binary=False)
mesh_quad1 = meshio.Mesh(mesh_points_global1, quads1)
out_quad1 = "Jordan_quad_global.vtk"
meshio.write(out_quad1, mesh_quad1, file_format="vtk", binary=False)

