import os
import numpy as np


def read_binary(file: str, num_read: int, size: int, endian: str = "little", signed: bool = False):
    """
    Reads integer values from binary files that are output of RSQSim

    :param file: file to read
    :param num_read: number of integers to read
    :param size: size of number to read (in bytes)
    :param endian: usually "little" unless we end up running on a non-standard system
    :param signed: include capacity for reading negative values (False if reading positive integers only)
    TODO: Could this be faster in cython?
    :return:
    """
    # Check that parameter supplied for endianness makes sense
    assert endian in ("little", "big"), "Must specify either 'big' or 'little' endian"
    assert os.path.exists(file)
    with open(file, "rb") as fid:
        # Container to store numbers as they are read
        number_list = []
        # Set counter to zero... indexing necessary because of the way RSQSim is set up
        count = 0
        # Read in required number of bytes
        byte = fid.read(size)
        # TODO: is there a better way of dealing with empty bytes?
        while count < num_read and byte != b"":
            # turn bytes into python integer
            byte_int = int.from_bytes(byte, byteorder=endian, signed=signed)
            # add to list of read integers
            number_list.append(byte_int)
            # Increase index and read bytes for next integer
            count += 1
            byte = fid.read(size)
    return number_list


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
