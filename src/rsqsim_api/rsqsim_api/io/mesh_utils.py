import numpy as np
import meshio
import glob
import os

expected_formats = ["stl", "obj", "e", "txt"]


def change_mesh_format(in_mesh: str, out_format: str, out_mesh: str = None):
    assert os.path.exists(in_mesh)
    assert out_format in expected_formats

    mesh = meshio.read(in_mesh)
    if out_mesh is not None:
        out_mesh_name = out_mesh
    else:
        handle = in_mesh.split(".")[0]
        out_mesh_name = handle + "." + out_format

    if out_format == "txt":
        mesh_as_array = mesh_to_array(mesh)
        np.savetxt(out_mesh_name, mesh_as_array, delimiter=" ", fmt="%.4f")

    else:
        mesh.write(out_mesh_name, file_format=out_format)


def mesh_to_array(mesh: meshio.Mesh):
    assert "triangle" in mesh.cells_dict.keys()
    triangles = mesh.cells_dict["triangle"]
    point_dict = {i: point for i, point in enumerate(mesh.points)}
    mesh_as_array = np.array([np.hstack([point_dict[vertex] for vertex in tri]) for tri in triangles])
    return mesh_as_array


def convert_multi(search_string: str, out_format: str):
    files = list(glob.glob(search_string))
    assert len(files) > 0
    for fname in files:
        change_mesh_format(fname, out_format)


def array_to_mesh(triangle_array: np.array):
    """
    Turns an array with 9 columns representing triangle vertices into a meshio mesh object
    :param triangle_array: X,Y,Z coordinates of each vertex of each triangle
    """
    assert triangle_array.shape[1] == 9
    # Create list of all vertices
    all_vertices = np.reshape(triangle_array, (int(triangle_array.shape[0] * 3), int(triangle_array.shape[1] / 3)))

    # Find unique vertices and assign them a number
    unique_vertices = np.unique(all_vertices, axis=0)
    vertex_dic = {tuple(vertex): i for i, vertex in enumerate(unique_vertices)}

    # Loop through triangles, building list of vertex numbers for each triangle
    tri_list = []
    for tri in triangle_array:
        tri_list.append([vertex_dic[tuple(vi)] for vi in tri.reshape((3, 3))])

    # Turn into a mesh object
    mesh = meshio.Mesh(points=unique_vertices, cells=[("triangle", tri_list)])

    return mesh


def tri_slip_rake_to_mesh(in_array: np.ndarray):
    """
    Turns an array with 9 columns representing triangle vertices into a meshio mesh object
    :param in_array: X,Y,Z coordinates of each vertex of each triangle, slip rate (in m/s) and rake (degrees)
    """
    # Check that mesh has 11 columns
    assert in_array.shape[1] == 11
    # Build mesh from first 9 columns
    mesh = array_to_mesh(in_array[:, :9])

    # Dictionary containing slip rate and rake as arrays
    data_dict = {}
    data_dict["slip_rate"] = in_array[:, 9]
    data_dict["rake"] = in_array[:, 10]

    # Add dictionary to mesh as cell data
    mesh.cell_data = data_dict

    return mesh


def quads_to_vtk(quads):
    vertices = np.unique(np.vstack([rect for rect in quads]), axis=0)
    vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
    new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in
                              quads]
    mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})
    return mesh