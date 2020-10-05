import ipyvolume as ipv

from rsqsim_api.containers.fault import RsqSimSegment, RsqSimTriangularPatch

def fault_to_ipv(fault: RsqSimSegment):
    x, y, z =