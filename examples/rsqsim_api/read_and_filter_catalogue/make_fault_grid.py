import os
import numpy as np
from rsqsim_api.containers.fault import RsqSimSegment
import rsqsim_api.io.write_utils as rwu

import meshio

import eq_fault_geom.faultmeshio.tsurf

# Locate files relative to script.
run_dir = os.path.dirname(__file__)
jordan_dxf = os.path.join(run_dir, "../../../data/cfm_dxf/Jordan.dxf")
wellington_tsurf = os.path.join(run_dir, "../../../data/cfm_tsurf/Wellington_Hutt_Valley_1.ts")
white_island_tsurf = os.path.join(run_dir, "../../../data/cfm_tsurf/White_Island_North_2_.ts")

# Read in Jordan fault.
fault1 = RsqSimSegment.from_dxf(jordan_dxf)
points1 = np.array(fault1.vertices)
edges1 = fault1.boundary
triangles1 = fault1.triangles

# Generate mesh.
resolution = 3000.0
fault_mesh_info1 = rwu.create_quad_mesh_from_fault(points1, edges1, triangles1, resolution=resolution)

triangles1 = [("triangle", triangles1)]

# Write VTK files in local and global coordinates.
mesh_tri_global1 = meshio.Mesh(points1, triangles1)
out_tri_global1 = "Jordan_tri_global.vtk"
meshio.write(out_tri_global1, mesh_tri_global1, file_format="vtk", binary=False)

mesh_tri_local1 = meshio.Mesh(fault_mesh_info1['points_local'], triangles1)
out_tri_local1 = "Jordan_tri_local.vtk"
meshio.write(out_tri_local1, mesh_tri_local1, file_format="vtk", binary=False)

quad_cells1 = rwu.create_cells_from_dims(fault_mesh_info1['num_horiz_points'], fault_mesh_info1['num_vert_points'])
quads1 = [("quad", quad_cells1)]

mesh_quad_global1 = meshio.Mesh(fault_mesh_info1['mesh_points_global'], quads1)
out_quad_global1 = "Jordan_quad_global.vtk"
meshio.write(out_quad_global1, mesh_quad_global1, file_format="vtk", binary=False)

mesh_quad_local1 = meshio.Mesh(fault_mesh_info1['mesh_points_local'], quads1)
out_quad_local1 = "Jordan_quad_local.vtk"
meshio.write(out_quad_local1, mesh_quad_local1, file_format="vtk", binary=False)

# Read in Wellington fault segment.
fault2 = eq_fault_geom.faultmeshio.tsurf.tsurf(wellington_tsurf)
points2 = fault2.mesh.points
triangles2 = fault2.mesh.cells_dict['triangle']
edge_inds2 = rwu.get_mesh_boundary(triangles2)
edges2 = points2[edge_inds2,:]

# Generate mesh.
fault_mesh_info2 = rwu.create_quad_mesh_from_fault(points2, edges2, triangles2, resolution=resolution)

triangles2 = [("triangle", triangles2)]

# Write VTK files in local and global coordinates.
mesh_tri_global2 = meshio.Mesh(points2, triangles2)
out_tri_global2 = "Wellington_Hutt_valley_1_tri_global.vtk"
meshio.write(out_tri_global2, mesh_tri_global2, file_format="vtk", binary=False)

mesh_tri_local2 = meshio.Mesh(fault_mesh_info2['points_local'], triangles2)
out_tri_local2 = "Wellington_Hutt_valley_1_tri_local.vtk"
meshio.write(out_tri_local2, mesh_tri_local2, file_format="vtk", binary=False)

quad_cells2 = rwu.create_cells_from_dims(fault_mesh_info2['num_horiz_points'], fault_mesh_info2['num_vert_points'])
quads2 = [("quad", quad_cells2)]

mesh_quad_global2 = meshio.Mesh(fault_mesh_info2['mesh_points_global'], quads2)
out_quad_global2 = "Wellington_Hutt_valley_1_quad_global.vtk"
meshio.write(out_quad_global2, mesh_quad_global2, file_format="vtk", binary=False)

mesh_quad_local2 = meshio.Mesh(fault_mesh_info2['mesh_points_local'], quads2)
out_quad_local2 = "Wellington_Hutt_valley_1_quad_local.vtk"
meshio.write(out_quad_local2, mesh_quad_local2, file_format="vtk", binary=False)

