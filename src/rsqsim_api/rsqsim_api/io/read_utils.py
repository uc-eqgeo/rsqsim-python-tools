"""Utilities for reading RSQSim output files in text, binary, CSV, DXF, STL, and VTK formats."""
import os

import meshio
import numpy as np
import pandas as pd
import ezdxf
from scipy.spatial import KDTree

catalogue_columns = ["t0", "m0", "mw", "x", "y", "z", "area", "dt"]


def read_text(file: str, format: str):
    """
    Read scalar values from a text file output by RSQSim.

    Produced when RSQSim is compiled with serial (text) output mode.

    Parameters
    ----------
    file :
        Path to the text file to read.
    format :
        Data type specifier: ``"d"`` for double (float64) or ``"i"``
        for integer (int32).

    Returns
    -------
    numpy.ndarray
        Flattened 1-D array of values read from the file.

    Raises
    ------
    AssertionError
        If ``format`` is not ``"d"`` or ``"i"``, or if ``file`` does not
        exist.
    """
    # Check file existence and that parameter supplied for format makes sense.
    assert format in ("d", "i")
    assert os.path.exists(file)
    if format == "d":
        numbers = np.loadtxt(file, dtype=np.float64).flatten()
    else:
        numbers = np.loadtxt(file, dtype=np.int32).flatten()

    return numbers

def read_binary(file: str, format: str, endian: str = "little"):
    """
    Read scalar values from a binary file output by RSQSim.

    Parameters
    ----------
    file :
        Path to the binary file to read.
    format :
        Data type specifier: ``"d"`` for double (float64) or ``"i"``
        for integer (int32).
    endian :
        Byte order of the file.  ``"little"`` (default) is standard for
        most modern x86 systems; use ``"big"`` for non-standard platforms.

    Returns
    -------
    numpy.ndarray
        Flattened 1-D array of values read from the file.

    Raises
    ------
    AssertionError
        If ``endian`` is not ``"little"`` or ``"big"``, if ``format`` is
        not ``"d"`` or ``"i"``, or if ``file`` does not exist.
    """
    # Check that parameter supplied for endianness makes sense
    assert endian in ("little", "big"), "Must specify either 'big' or 'little' endian"
    endian_sign = "<" if endian == "little" else ">"
    assert format in ("d", "i")
    assert os.path.exists(file)
    if format == "d":
        numbers = np.fromfile(file, endian_sign + "f8").flatten()
    else:
        numbers = np.fromfile(file, endian_sign + "i4").flatten()

    return numbers


def read_csv_and_array(prefix: str, read_index: bool = True):
    """
    Read a catalogue CSV and its associated NumPy array files from a common prefix.

    Expects files named ``<prefix>_catalogue.csv``, ``<prefix>_events.npy``,
    ``<prefix>_patches.npy``, ``<prefix>_slip.npy``, and
    ``<prefix>_slip_time.npy`` to exist on disk.

    Parameters
    ----------
    prefix :
        File path prefix (with or without trailing underscore).
    read_index : bool, optional
        If ``True`` (default), read the first column of the CSV as the
        DataFrame index.

    Returns
    -------
    list
        ``[df, events, patches, slip, slip_time]`` where ``df`` is a
        ``pandas.DataFrame`` and the remaining elements are NumPy arrays
        loaded from the corresponding ``.npy`` files.

    Raises
    ------
    AssertionError
        If ``prefix`` is an empty string.
    FileNotFoundError
        If any of the five expected files is missing.
    """
    assert prefix, "Empty prefix string supplied"
    if prefix[-1] != "_":
        prefix += "_"
    suffixes = ["catalogue.csv", "events.npy", "patches.npy", "slip.npy", "slip_time.npy"]
    file_list = [prefix + suffix for suffix in suffixes]
    for file, suffix in zip(file_list, suffixes):
        if not os.path.exists(file):
            raise FileNotFoundError("{} file missing!".format(suffix))
    if read_index:
        df = pd.read_csv(file_list[0], index_col=0)
    else:
        df = pd.read_csv(file_list[0])
    array_ls = [np.load(file) for file in file_list[1:]]

    return [df] + array_ls



def read_earthquakes(earthquake_file: str, get_patch: bool = False, eq_start_index: int = None,
                     eq_end_index: int = None, endian: str = "little"):
    """
    Read earthquake event data, inferring companion file names from the event file prefix.

    Based on R scripts by Keith Richards-Dinger.  The earthquake file is
    expected to follow the naming convention ``eqs.<prefix>.out``.

    Parameters
    ----------
    earthquake_file :
        Path to the RSQSim earthquake output file, usually with a
        ``.out`` suffix and named ``eqs.<prefix>.out``.
    get_patch :
        If ``True``, also read per-patch rupture data.  Currently unused
        placeholder.
    eq_start_index :
        Zero-based index of the first earthquake to read.  Both
        ``eq_start_index`` and ``eq_end_index`` must be provided together.
    eq_end_index :
        Zero-based index of the last earthquake to read (exclusive).
    endian :
        Byte order for binary companion files.  Defaults to ``"little"``.

    Raises
    ------
    AssertionError
        If ``endian`` is not ``"little"`` or ``"big"``, or if
        ``earthquake_file`` does not exist.
    ValueError
        If ``eq_start_index >= eq_end_index`` when both are provided.
    """
    assert endian in ("little", "big"), "Must specify either 'big' or 'little' endian"
    assert os.path.exists(earthquake_file)
    if not any([a is None for a in (eq_start_index, eq_end_index)]):
        if eq_start_index >= eq_end_index:
            raise ValueError("eq_start index should be smaller than eq_end_index!")

    # Get full path to file and working directory
    abs_file_path = os.path.abspath(earthquake_file)
    file_base_name = os.path.basename(abs_file_path)

    # Get file prefix from basename
    split_by_dots = file_base_name.split(".")
    # Check that filename fits expected format
    if not all([split_by_dots[0] == "eqs", split_by_dots[-1] == "out"]):
        print("Warning: non-standard file name.")
        print("Expecting earthquake file name to have the format: eqs.{prefix}.out")
        print("using 'catalogue' as prefix...")
        prefix = "catalogue"
    else:
        # Join prefix back together if necessary, warning if empty
        prefix_list = split_by_dots[1:-1]
        if len(prefix_list) == 1:
            prefix = prefix_list[0]
            if prefix.strip() == "":
                print("Warning: empty prefix string")
        else:
            prefix = ".".join(*prefix_list)

    # Search for binary files in directory
    tau_file = abs_file_path + "/tauDot.{}.out".format(prefix)
    sigmat_file = abs_file_path + "/sigmaDot.{}.out".format(prefix)


def read_earthquake_catalogue(catalogue_file: str):
    """
    Read an RSQSim earthquake catalogue text file into a DataFrame.

    Parses the catalogue file produced by RSQSim, skipping the header
    block that ends with the line ``%%% end input files``, and loads the
    subsequent rows into a DataFrame with the standard catalogue columns
    ``t0, m0, mw, x, y, z, area, dt``.

    Parameters
    ----------
    catalogue_file :
        Path to the RSQSim catalogue text file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``["t0", "m0", "mw", "x", "y", "z",
        "area", "dt"]``, one row per earthquake event.

    Raises
    ------
    AssertionError
        If ``catalogue_file`` does not exist.
    """
    assert os.path.exists(catalogue_file)

    with open(catalogue_file, "r") as fid:
        data = fid.readlines()

    start_eqs = data.index("%%% end input files\n") + 1
    data_array = np.loadtxt(data[start_eqs:])
    earthquake_catalogue = pd.DataFrame(data_array[:, :8], columns=catalogue_columns)
    return earthquake_catalogue




# def read_fault(fault_file_name: str, check_if_grid: bool = True, )

def read_ts_coords(filename):
    """
    Read vertex and triangle data from a tsurf (``.ts``) file.

    Parses the SCEC Community Fault Model tsurf format, extracting
    vertex coordinates and triangle connectivity.  Based on the MATLAB
    script ``ReadAndSaveCfm.m`` by Brendan Meade.

    Parameters
    ----------
    filename :
        Path to the tsurf ``.ts`` file.

    Returns
    -------
    vrtx : numpy.ndarray of shape (n_vertices, 4)
        Vertex data array.  Each row is ``[vertex_id, x, y, z]``.
    trgl : numpy.ndarray of shape (n_triangles, 3), dtype int
        Triangle connectivity array.  Each row gives the three vertex IDs
        forming a triangle.
    tri : numpy.ndarray of shape (n_triangles, 9)
        Triangle data array.  Each row contains the (x, y, z) coordinates
        of all three corners concatenated: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.

    Notes
    -----
    Copyright Paul Kaeufl, July 2014.
    Original MATLAB script:
    http://structure.rc.fas.harvard.edu/cfm/download/meade/ReadAndSaveCfm.m
    """

    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    idxVrtx = [idx for idx, l in enumerate(lines)
               if 'VRTX' in l or 'PVRTX' in l]
    idxTrgl = [idx for idx, l in enumerate(lines) if 'TRGL' in l]
    nVrtx = len(idxVrtx)
    nTrgl = len(idxTrgl)
    vrtx = np.zeros((nVrtx, 4))
    trgl = np.zeros((nTrgl, 3), dtype='int')
    tri = np.zeros((nTrgl, 9))
    for k, iVrtx in enumerate(idxVrtx):
        line = lines[iVrtx]
        tmp = line.split()
        vrtx[k] = [int(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4])]

    for k, iTrgl in enumerate(idxTrgl):
        line = lines[iTrgl]
        tmp = line.split(' ')
        trgl[k] = [int(tmp[1]), int(tmp[2]), int(tmp[3])]
        for l in range(3):
            i1 = l * 3
            i2 = 3 * (l + 1)
            vertex_i = vrtx[vrtx[:, 0] == trgl[k, l]][0]
            tri[k, i1:i2] = vertex_i[1:]
    return vrtx, trgl, tri


def read_dxf(dxf_file: str):
    """
    Read a triangulated mesh and boundary polyline from a DXF file.

    Expects a DXF file exported from Move containing exactly one
    ``POLYLINE`` boundary and one or more ``3DFACE`` triangles.

    Parameters
    ----------
    dxf_file :
        Path to the DXF file.

    Returns
    -------
    triangle_array : numpy.ndarray of shape (n_triangles, 9)
        Each row contains the (x, y, z) coordinates of the three triangle
        corners concatenated: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.
    boundary_array : numpy.ndarray of shape (n_points, 3)
        (x, y, z) coordinates of the boundary polyline vertices.

    Raises
    ------
    AssertionError
        If ``dxf_file`` does not exist, or if the file does not contain
        both ``3DFACE`` and ``POLYLINE`` entities.
    ValueError
        If the file contains more than one ``POLYLINE`` boundary.
    """
    assert os.path.exists(dxf_file)
    dxf = ezdxf.readfile(dxf_file)
    msp = dxf.modelspace()
    dxftypes = [e.dxftype() for e in msp]
    assert all([a in dxftypes for a in ("3DFACE", "POLYLINE")]), "{}: Expected triangles and boundary".format(dxf_file)
    if dxftypes.count("POLYLINE") > 1:
        raise ValueError("{}: Too many boundaries lines...".format(dxf_file))


    triangle_ls = []
    boundary_array = None
    for entity in msp:
        if entity.dxftype() == "3DFACE":
            triangle = np.array([vertex.xyz for vertex in entity])
            unique_triangle = np.unique(triangle, axis=0).reshape((9,))
            triangle_ls.append(unique_triangle)

        elif entity.dxftype() == "POLYLINE":
            boundary_ls = []
            for point in entity.points():
                boundary_ls.append(point.xyz)
            boundary_array = np.array(boundary_ls)

    triangle_array = np.array(triangle_ls)

    return triangle_array, boundary_array


def read_stl(stl_file: str, min_point_sep=100.):
    """
    Read a triangulated surface mesh from an STL file.

    Merges near-duplicate vertices (within ``min_point_sep``) by averaging
    their positions and removing degenerate triangles that result from the
    merge.

    Parameters
    ----------
    stl_file :
        Path to the STL mesh file.
    min_point_sep :
        Minimum separation in metres below which two vertices are
        considered duplicates and merged.  Defaults to 100.0 m.

    Returns
    -------
    numpy.ndarray of shape (n_triangles, 9)
        Each row contains the (x, y, z) coordinates of the three triangle
        corners: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.

    Raises
    ------
    AssertionError
        If ``stl_file`` does not exist or if the mesh contains no
        triangular cells.
    """
    assert os.path.exists(stl_file)

    mesh = meshio.read(stl_file)

    assert "triangle" in mesh.cells_dict.keys()
    triangles = mesh.cells_dict["triangle"]
    mesh_points = mesh.points[:]
    point_tree = KDTree(mesh.points)
    distances, indices = point_tree.query(mesh.points, k=[2])
    problem_indices = indices[distances < min_point_sep]
    paired_indices = []
    for i1 in problem_indices:
        i2 = indices[i1][0]
        if not any([i in paired_indices for i in [i1, i2]]):
            p1 = mesh.points[i1]
            p2 = mesh.points[i2]
            mesh_points[i2] = 0.5 * (p1 + p2)
            mesh_points[i1] = 0.5 * (p1 + p2)
            triangles[triangles == i1] = i2
            paired_indices += [i1, i2]
    if len(problem_indices) > 0:
        num_tris_pre = len(triangles)
        tris_num_unique_vertices = np.unique(triangles, axis=1)
        tri_lens = np.array([len(np.unique(tri)) for tri in tris_num_unique_vertices])
        valid_tri_indices = np.where(tri_lens == 3)[0]
        triangles = triangles[valid_tri_indices]
        num_tris_post = len(triangles)
        if num_tris_post < num_tris_pre:
            print("Warning: {} triangles removed from mesh due to duplicate vertices".format(
                num_tris_pre - num_tris_post))
    point_dict = {i: point for i, point in enumerate(mesh_points)}
    mesh_as_array = np.array([np.hstack([point_dict[vertex] for vertex in tri]) for tri in triangles])
    return mesh_as_array

def read_vtk(vtk_file: str, min_point_sep=1.):
    """
    Read a triangulated surface mesh with slip and rake data from a VTK file.

    Merges near-duplicate vertices (within ``min_point_sep``) and removes
    degenerate triangles, preserving the associated slip and rake cell data.

    Parameters
    ----------
    vtk_file :
        Path to the VTK mesh file.
    min_point_sep :
        Minimum separation in metres below which two vertices are
        considered duplicates and merged.  Defaults to 1.0 m.

    Returns
    -------
    mesh_as_array : numpy.ndarray of shape (n_triangles, 9)
        Each row contains the (x, y, z) coordinates of the three triangle
        corners: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3]``.
    slip : numpy.ndarray of shape (n_triangles,)
        Slip magnitude (metres) for each triangle after duplicate removal.
    rake : numpy.ndarray of shape (n_triangles,)
        Rake angle (degrees) for each triangle after duplicate removal.

    Raises
    ------
    AssertionError
        If ``vtk_file`` does not exist, if the mesh contains no triangular
        cells, or if the ``slip`` and ``rake`` cell data arrays are absent.
    """
    assert os.path.exists(vtk_file)

    mesh = meshio.read(vtk_file)

    assert "triangle" in mesh.cells_dict.keys()
    assert all([data in mesh.cell_data.keys() for data in ("slip", "rake")])
    triangles = mesh.cells_dict["triangle"]
    mesh_points = mesh.points[:]
    point_tree = KDTree(mesh.points)
    distances, indices = point_tree.query(mesh.points, k=[2])
    problem_indices = indices[distances < min_point_sep]
    paired_indices = []
    for i1 in problem_indices:
        i2 = indices[i1][0]
        if not any([i in paired_indices for i in [i1, i2]]):
            p1 = mesh.points[i1]
            p2 = mesh.points[i2]
            mesh_points[i2] = 0.5 * (p1 + p2)
            mesh_points[i1] = 0.5 * (p1 + p2)
            triangles[triangles == i1] = i2
            paired_indices += [i1, i2]
    if len(problem_indices) > 0:
        num_tris_pre = len(triangles)
        tris_num_unique_vertices = np.unique(triangles, axis=1)
        tri_lens = np.array([len(np.unique(tri)) for tri in tris_num_unique_vertices])
        valid_tri_indices = np.where(tri_lens == 3)[0]
        triangles = triangles[valid_tri_indices]
        slip = mesh.cell_data["slip"][0][valid_tri_indices]
        rake = mesh.cell_data["rake"][0][valid_tri_indices]
        num_tris_post = len(triangles)
        if num_tris_post < num_tris_pre:
            print("Warning: {} triangles removed from mesh due to duplicate vertices".format(num_tris_pre - num_tris_post))
    else:
        slip = mesh.cell_data["slip"][0]
        rake = mesh.cell_data["rake"][0]
    point_dict = {i: point for i, point in enumerate(mesh_points)}
    mesh_as_array = np.array([np.hstack([point_dict[vertex] for vertex in tri]) for tri in triangles])


    return mesh_as_array, slip, rake
