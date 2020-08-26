import os
import numpy as np
import pandas as pd
import struct

catalogue_columns = ["t0", "m0", "mw", "x", "y", "z", "area", "dt"]

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
    with open(file, "rb") as fid:
        data = fid.read()
        if format == "d":
            numbers = np.array([struct.unpack(endian_sign + "d", data[i:i+8], ) for i in np.arange(0, len(data),
                                                                                                   8)]).flatten()
        else:
            numbers = np.array([struct.unpack(endian_sign + "i", data[i:i+4], ) for i in np.arange(0, len(data),
                                                                                                   4)]).flatten()

    return numbers


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
