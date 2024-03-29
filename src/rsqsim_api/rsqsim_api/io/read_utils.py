import os

import meshio
import numpy as np
import pandas as pd
import ezdxf
from scipy.spatial import KDTree

catalogue_columns = ["t0", "m0", "mw", "x", "y", "z", "area", "dt"]


def read_text(file: str, format: str):
    """
    Reads scalar values from text files that are output of RSQSim.
    Note that these files are produced if the code is compiled with
    serial output.
    :param file: file to read
    :param format: either "d" (double) or "i" (integer)
    :return:
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
    Reads integer values from binary files that are output of RSQSim

    :param file: file to read
    :param format: either "d" (double) or "i" (integer)
    :param endian: usually "little" unless we end up running on a non-standard system
    :return:
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
    Reads earthquakes, inferring list file names from prefix of earthquake file.
    Based on R scripts by Keith Richards-Dinger.

    :param earthquake_file: usually has a ".out" suffix
    :param get_patch:
    :param eq_start_index:
    :param eq_end_index:
    :param endian:
    :return:
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
    This script reads in the tsurf (*.ts) files for the SCEC Community Fault Model (cfm)
    as a numpy array.
    The script is based on the matlab script ReadAndSaveCfm.m by Brendan Meade available
    from http://structure.rc.fas.harvard.edu/cfm/download/meade/ReadAndSaveCfm.m
    Copyright Paul Kaeufl, July 2014
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
    Reads mesh and boundary from dxf file exported from move. Returns boundary (as array) and triangles
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

