import os
import numpy as np
from rsqsim_api.containers.fault import RsqSimSegment
from rsqsim_api.io.write_utils import fit_plane_to_points

# Locate DXF file relative to script
run_dir = os.path.dirname(__file__)
jordan_dxf = os.path.join(run_dir, "../../../data/cfm_dxf/Jordan.dxf")

# Read in DXF file
fault1 = RsqSimSegment.from_dxf(jordan_dxf)

# Get points and compute normal.
points1 = np.array(fault1.vertices)
(plane_normal1, plane_origin1) = fit_plane_to_points(points1)

print(plane_normal1, plane_origin1)

# Test case where fault is vertical.
a = 0.5
b = 0.5
d = 5000.0
num_points = 200
rng = np.random.default_rng()
y = 1000.0*rng.random(num_points, dtype=np.float64) - 500.0
z = 1000.0*rng.random(num_points, dtype=np.float64) - 500.0
x = (-b*y - d)/a
points2 = np.column_stack((x, y, z))
(plane_normal2, plane_origin2) = fit_plane_to_points(points2)

print(plane_normal2, plane_origin2)
