"""Utilities for converting between triangulated mesh formats and NumPy arrays."""
import numpy as np
import meshio
import glob
import os

expected_formats = ["stl", "obj", "e", "txt"]


def change_mesh_format(in_mesh: str, out_format: str, out_mesh: str = None):
    """
    Convert a mesh file from one format to another.

    Reads the input mesh with meshio and writes it in the requested output
    format.  If ``out_format`` is ``"txt"``, the mesh is converted to a
    flat NumPy array and saved with ``numpy.savetxt``.

    Parameters
    ----------
    in_mesh :
        Path to the input mesh file (any format supported by meshio).
    out_format :
        Target format.  Must be one of ``"stl"``, ``"obj"``, ``"e"``,
        or ``"txt"``.
    out_mesh :
        Output file path.  If ``None``, the input file stem is reused
        with the new extension.

    Raises
    ------
    AssertionError
        If ``in_mesh`` does not exist or ``out_format`` is not in the
        allowed list.
    """
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
    """
    Convert a meshio Mesh to a flat NumPy triangle array.

    Expands the indexed triangle representation into a dense array where
    each row contains the (x, y, z) coordinates of all three corner
    vertices of one triangle.

    Parameters
    ----------
    mesh :
        meshio Mesh object containing at least one set of triangular cells.

    Returns
    -------
    numpy.ndarray of shape (n_triangles, 9)
        Each row is ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.

    Raises
    ------
    AssertionError
        If the mesh contains no triangular cells.
    """
    assert "triangle" in mesh.cells_dict.keys()
    triangles = mesh.cells_dict["triangle"]
    point_dict = {i: point for i, point in enumerate(mesh.points)}
    mesh_as_array = np.array([np.hstack([point_dict[vertex] for vertex in tri]) for tri in triangles])
    return mesh_as_array


def convert_multi(search_string: str, out_format: str):
    """
    Convert all mesh files matching a glob pattern to a target format.

    Parameters
    ----------
    search_string :
        Glob pattern used to find input mesh files, e.g.
        ``"/path/to/meshes/*.stl"``.
    out_format :
        Target format passed to :func:`change_mesh_format`.

    Raises
    ------
    AssertionError
        If no files match ``search_string``.
    """
    files = list(glob.glob(search_string))
    assert len(files) > 0
    for fname in files:
        change_mesh_format(fname, out_format)


def array_to_mesh(triangle_array: np.array):
    """
    Convert a flat triangle array to a meshio Mesh object.

    Deduplicates the vertices across all triangles and builds an indexed
    meshio Mesh with a single set of triangular cells.

    Parameters
    ----------
    triangle_array : numpy.ndarray of shape (n_triangles, 9)
        Each row contains the (x, y, z) coordinates of the three corner
        vertices of one triangle: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.

    Returns
    -------
    meshio.Mesh
        Mesh with unique ``points`` (shape ``(n_unique_verts, 3)``) and
        triangular cells referencing them by index.

    Raises
    ------
    AssertionError
        If ``triangle_array`` does not have exactly 9 columns.
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
    Convert a triangle array with slip and rake columns to a meshio Mesh.

    Builds a mesh from the first 9 columns (vertex coordinates) and
    attaches the slip rate and rake columns as cell data.

    Parameters
    ----------
    in_array : numpy.ndarray of shape (n_triangles, 11)
        Columns 0–8 are the (x,y,z) coordinates of the three triangle
        corners (as in :func:`array_to_mesh`).  Column 9 is slip rate
        in m/s and column 10 is rake in degrees.

    Returns
    -------
    meshio.Mesh
        Mesh with triangular cells and ``cell_data`` containing
        ``"slip_rate"`` and ``"rake"`` arrays.

    Raises
    ------
    AssertionError
        If ``in_array`` does not have exactly 11 columns.
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
    """
    Convert a list of quadrilateral patches to a meshio Mesh for VTK export.

    Deduplicates vertices across all quads and builds an indexed meshio
    Mesh with quad cells.

    Parameters
    ----------
    quads :
        Iterable of quadrilateral patches.  Each element should be an
        array-like of shape (4, 3) giving the four corner (x, y, z)
        coordinates in NZTM (metres).

    Returns
    -------
    meshio.Mesh
        Mesh with unique ``points`` and ``"quad"`` cells.
    """
    vertices = np.unique(np.vstack([rect for rect in quads]), axis=0)
    vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
    new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in
                              quads]
    mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})
    return mesh
