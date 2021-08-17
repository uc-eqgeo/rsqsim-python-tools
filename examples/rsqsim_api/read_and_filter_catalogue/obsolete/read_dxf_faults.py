import os
from rsqsim_api.fault.segment import RsqSimSegment

# Locate DXF file relative to script
run_dir = os.path.dirname(__file__)
jordan_dxf = os.path.join(run_dir, "../../../../data/cfm_dxf/Jordan.dxf")

# Read in DXF file
fault = RsqSimSegment.from_dxf(jordan_dxf)

print(fault.boundary, fault.triangles, fault.vertices)
