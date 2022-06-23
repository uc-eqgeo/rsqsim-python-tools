import numpy as np
from rsqsim_api.fault.multifault import RsqSimMultiFault
import os.path
import fnmatch
import meshio

data_dir = "../../../data/shaw2021/rundir5382"

faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(data_dir, "zfault_Deepen.in"),
                                                os.path.join(data_dir, "znames_Deepen.in"), transform_from_utm=True)

baseFault = "wellington"
faults2select = [name for name in faults.names if fnmatch.fnmatch(name, baseFault+"*")]
fault_selection = RsqSimMultiFault([faults.name_dic[name] for name in faults2select])

fault0 = fault_selection.faults[0]

wellington_names = set([name[:-1] for name in fault_selection.names])
for name in list(wellington_names):
    new_fault = faults.merge_segments(name, fault_name=name)
    dip_angle = new_fault.get_average_dip()
    print(name, dip_angle, new_fault.dip_dir)
    new_fault.to_vtk("fault_vtks/" + name + ".vtk")
    new_fault_rect = new_fault.discretize_rectangular_tiles(tile_size=5000.)
    vertices = np.unique(np.vstack([rect for rect in new_fault_rect]), axis=0)
    vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
    new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in new_fault_rect]
    mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})
    meshio.write("fault_vtks/" + name + "_quad.vtk", mesh)

# tara = faults.merge_segments("wellington:tararua3", fault_name="wellington:tararua3")
