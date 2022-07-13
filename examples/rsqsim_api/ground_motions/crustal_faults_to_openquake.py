import numpy as np
from rsqsim_api.fault.multifault import RsqSimMultiFault
import os.path
import fnmatch
import meshio

data_dir = "../../../data/shaw2021/rundir5382"

faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(data_dir, "zfault_Deepen.in"),
                                                os.path.join(data_dir, "znames_Deepen.in"), transform_from_utm=True)



# Find Hikurangi segments
hikurangi_names = [name for name in faults.names if fnmatch.fnmatch(name, "*hikurangi*")]
# Merge into one big fault
hik_merged = faults.merge_segments("*hikurangi*", fault_name="hikurangi")
# Average dip
dip_angle = hik_merged.get_average_dip(15000.)
print("hik_merged", dip_angle, hik_merged.dip_dir)
# Discretize into rectangular tiles
new_fault_rect = hik_merged.discretize_rectangular_tiles(tile_size=15000.)

# Turn into quadrilateral mesh
vertices = np.unique(np.vstack([rect for rect in new_fault_rect]), axis=0)
vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in new_fault_rect]
mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})
meshio.write("hik_merged" + ".vtk", mesh)

# As for Hikurangi, but for Puysegur
puy_names = [name for name in faults.names if fnmatch.fnmatch(name, "*puysegar*")]
puy_merged = faults.merge_segments("*puysegar*", fault_name="puysegar")
dip_angle = puy_merged.get_average_dip(15000.)
print("puysegar_merged", dip_angle, puy_merged.dip_dir)
new_fault_rect = puy_merged.discretize_rectangular_tiles(tile_size=15000.)
vertices = np.unique(np.vstack([rect for rect in new_fault_rect]), axis=0)
vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in new_fault_rect]
mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})
meshio.write("puy_merged" + ".vtk", mesh)


# Find all other faults
faults2select = [name for name in faults.names if not any([name in list_i for list_i in [hikurangi_names, puy_names]])]
fault_selection = RsqSimMultiFault([faults.name_dic[name] for name in faults2select])

if not os.path.exists("fault_vtks"):
    os.mkdir("fault_vtks")

# Find unique names without segment numbers
other_names = set([name[:-1] for name in fault_selection.names])
for name in list(other_names):
    # Attempt to discretize
    try:
        new_fault = faults.merge_segments(name, fault_name=name)
        dip_angle = new_fault.get_average_dip()
        # print(name, dip_angle, new_fault.dip_dir)
        new_fault.to_vtk("fault_vtks/" + name + ".vtk")
        # Smaller tile size than subduction zones
        new_fault_rect = new_fault.discretize_rectangular_tiles(tile_size=5000.)
        # Turn into quadrilateral mesh
        vertices = np.unique(np.vstack([rect for rect in new_fault_rect]), axis=0)
        vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
        new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in new_fault_rect]
        mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})
        meshio.write("fault_vtks/" + name + "_quad.vtk", mesh)
    except:
        print("Failed to merge", name)
