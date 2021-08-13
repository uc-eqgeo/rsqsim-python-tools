import geopandas as gpd
from shapely.geometry import LineString
import os




def make_journal_file(line: LineString, outfile, default_z = 0., depth=5.e4):
    out_str = ""
    for coord in list(line.coords):
        print(coord)
        x, y = coord[:2]
        out_str += f"cubit.cmd('create vertex location {x:.4f} {y:.4f} {default_z:.2f}')\n"

    vertex_numbers = ",".join([str(i+1) for i in range(len(list(line.coords)))])
    out_str += f"cubit.cmd('create surface vertex location {vertex_numbers}')\n"
    out_str += f"cubit.cmd('sweep surface 1 vector 0 0 -1 distance {depth:.2f}')\n"
    with open(outfile, "w") as out_id:
        out_id.write(out_str)

def make_journal_file_commands(line: LineString, outfile, outmesh: str, default_z = 0., depth=5.e4):
    out_str = ""
    for coord in list(line.coords):
        print(coord)
        x, y = coord[:2]
        out_str += f"create vertex location {x:.4f} {y:.4f} {default_z:.2f}\n"

    vertex_numbers = ",".join([str(i+1) for i in range(len(list(line.coords)) - 1)])
    surface_numbers = ",".join([str(i+1) for i in range(len(list(line.coords)) + 1)])
    out_str += f"create surface vertex {vertex_numbers}\n"
    out_str += f"sweep surface 1 vector 0 0 -1 distance {depth:.2f}\n"
    out_str += f"surface {surface_numbers} scheme trimesh geometry approximation angle 15\n"
    out_str += f"trimesher surface gradation 1.3\n"
    out_str += f"trimesher geometry sizing on\n"

    out_str += f"mesh surface {surface_numbers}\n"
    out_str += f"""export stl "{outmesh}" overwrite\n"""
    out_str += f"""exit()\n"""

    with open(outfile, "w") as out_id:
        out_id.write(out_str)




if __name__ == "__main__":
    data = "../../../examples/rsqsim_api/remeshing/kermadec_louisville_clipped.shp"
    gdf = gpd.read_file(data)
    out_mesh = "/home/UOCNT/arh128/PycharmProjects/rnc2-uc/examples/rsqsim_api/remeshing/offshore_limit.stl"
    out_obj = "/home/UOCNT/arh128/PycharmProjects/rnc2-uc/examples/rsqsim_api/remeshing/offshore_limit.obj"
    geom = gdf.geometry[0]
    exterior = geom.exterior
    journal = "test_vol.jou"
    make_journal_file_commands(exterior, journal, out_mesh, depth=2.e4)
    os.system(f"/opt/Trelis-17.1/bin/trelis -batch -nographics {journal}")




