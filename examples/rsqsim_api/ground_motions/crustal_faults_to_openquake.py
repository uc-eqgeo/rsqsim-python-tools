import numpy as np
from rsqsim_api.fault.multifault import RsqSimMultiFault
import os.path
import fnmatch
import meshio
import geopandas as gpd

data_dir = "/home/UOCNT/cpe88/PycharmProjects/rsqsim-python-tools/data/shaw_new_catalogue/NewZealand/rundir5382"

faults = RsqSimMultiFault.read_fault_file_bruce(main_fault_file=os.path.join(data_dir, "zfault_Deepen.in"),
                                                     name_file=os.path.join(data_dir, "znames_Deepen.in"),
                                                     transform_from_utm=True)



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
cfm = gpd.read_file("/home/UOCNT/cpe88/Data/CFM/cfm1_0.gpkg")
cfm_names=[name.lower() for name in cfm['Name']]
nearest_cfm = [difflib.get_close_matches(name[:-1], cfm_names, n=1) for name in fault_selection.names]
name_dict=dict(zip(fault_selection.names,nearest_cfm))
for key in name_dict.keys():
    if not bool(name_dict[key]):
        name_dict[key]=key[:-1].replace(" ","")
    else:
        name_dict[key]=name_dict[key][0].replace(" ","")
name_dict['wairau20']='wairau'
name_dict['wairau30']='wairau'

other_names = set(name_dict.values())
for name in list(other_names):
    # Attempt to discretize
    try:
        new_fault = faults.merge_segments(name,name_dict=name_dict,  fault_name=name)
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
