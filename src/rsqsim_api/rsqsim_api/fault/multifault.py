import glob
import os
from collections import Iterable
from typing import Union
import fnmatch

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rsqsim_api.visualisation.utilities import plot_coast
from rsqsim_api.fault.segment import RsqSimSegment


def check_unique_vertices(vertex_array: np.ndarray, tolerance: Union[int, float] = 1):
    """
    Efficiently checks whether vertices (3D point coordinates) may be duplicates by (1) sorting them,
    ans (2) finding distances between adjacent points in the sorted array
    :param vertex_array: Numpy array with 3 columns representing 3D vertex coordinates
    :param tolerance: distance (in metres) below which points are reported as possible duplicates
    :return:
    """
    assert isinstance(vertex_array, np.ndarray)

    assert vertex_array.shape[1] == 3
    sorted_vertices = np.sort(vertex_array, axis=0)
    differences = sorted_vertices[1:] - sorted_vertices
    distances = np.linalg.norm(differences, axis=1)

    num_closer = len(distances[np.where(distances <= tolerance)])

    return num_closer, float(tolerance)


class RsqSimMultiFault:
    def __init__(self, faults: Union[list, tuple, set]):
        self._faults = None
        self.faults = faults
        self._names = None
        self._name_dic = None
        self.patch_dic = {}
        for fault in self.faults:
            if self.patch_dic is not None:
                self.patch_dic.update(fault.patch_dic)

        self.faults_with_patches = {patch_num: patch.segment for patch_num, patch in self.patch_dic.items()}

    def filter_faults_by_patch_numbers(self, patch_ls: Union[int, list, tuple, np.ndarray]):
        if isinstance(patch_ls, int):
            return self.patch_dic[patch_ls]
        else:
            assert isinstance(patch_ls, (tuple, list, np.ndarray))
            assert patch_ls
            assert all([isinstance(x, int) for x in patch_ls])
            fault_ls = list(set([self.patch_dic[patch_number].segment for patch_number in patch_ls]))
            return RsqSimMultiFault(fault_ls)

    @property
    def faults(self):
        return self._faults

    @faults.setter
    def faults(self, faults: Union[list, tuple, set]):
        assert all([isinstance(fault, (RsqSimMultiFault, RsqSimSegment)) for fault in faults])
        self._faults = faults

    @property
    def names(self):
        if self._names is None:
            self.get_names()
        return self._names

    @property
    def name_dic(self):
        if self._name_dic is None:
            self.get_names()
        return self._name_dic

    def get_names(self):
        assert self.faults is not None
        self._names = [fault.name for fault in self.faults]
        self._name_dic = {fault.name: fault for fault in self.faults}

    @property
    def bounds(self):
        x0 = min([fault.bounds[0] for fault in self.faults])
        y0 = min([fault.bounds[1] for fault in self.faults])
        x1 = max([fault.bounds[2] for fault in self.faults])
        y1 = max([fault.bounds[3] for fault in self.faults])

        return np.array([x0, y0, x1, y1])

    @classmethod
    def read_fault_file(cls, fault_file: str, verbose: bool = False):
        """
        Read in opensha xml file?
        TODO: decide whether this is necessary, and if so finish it
        :param fault_file:
        :param verbose:
        :return:
        """
        assert os.path.exists(fault_file)
        # Read first 10 lines of file

    @classmethod
    def from_fault_file_keith(cls, fault_file: str, verbose: bool = False):
        """
        Read in an RSQSim fault file written according to Keith Richards-Dinger's convention.
        :param fault_file: Path to fault file
        :param verbose: Spit out more info if desired
        :return:
        """
        assert os.path.exists(fault_file)

        # Prepare info (types and headers) about columns
        column_dtypes = [float] * 11 + [int] + ["U50"]
        column_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                        "slip_rate", "fault_num", "fault_name"]

        # Read in data
        data = np.genfromtxt(fault_file, dtype=column_dtypes, names=column_names).T
        all_fault_nums = np.unique(data["fault_num"])
        num_faults = len(all_fault_nums)
        all_fault_names = np.unique(data["fault_name"])

        if not len(all_fault_names) == num_faults:
            print("Warning: not every fault has a corresponding name")

        # Populate faults with triangular patches
        patch_start = 0
        segment_ls = []

        for number in all_fault_nums:
            fault_data = data[data["fault_num"] == number]
            # Check that fault number has only one name associated with it
            associated_names = np.unique(fault_data["fault_name"])

            fault_name = associated_names[0] if associated_names.size > 0 else None

            if len(associated_names) > 1:
                print("More than one name provided for fault {:d}".format(number))
                print("Proceeding with first in list: {}".format(fault_name))
            if verbose:
                print("Reading fault: {:d}/{:d}: {}".format(number, num_faults, fault_name))

            # Extract data relevant for making triangles
            triangles = np.array([fault_data[name] for name in column_names[:9]]).T
            num_triangles = triangles.shape[0]
            # Reshape for creation of triangular patches
            all_vertices = triangles.reshape(triangles.shape[0] * 3, int(triangles.shape[1] / 3))

            patch_numbers = np.arange(patch_start, patch_start + num_triangles)

            if verbose:
                unique_vertices = np.unique(all_vertices, axis=0)
                num_closer, tolerance = check_unique_vertices(unique_vertices)
                if num_closer > 0:
                    print("{:d} points are closer than {:.2f} m together: duplicates?".format(num_closer, tolerance))

            # Create fault object
            fault_i = RsqSimSegment.from_triangles(triangles=triangles, patch_numbers=patch_numbers,
                                                   segment_number=number, fault_name=fault_name)

            segment_ls.append(fault_i)
            patch_start += num_triangles

        multi_fault = cls(segment_ls)

        return multi_fault

    @classmethod
    def read_fault_file_bruce(cls, main_fault_file: str, name_file: str, transform_from_utm: bool = False, from_pickle: bool = False):
        assert all([os.path.exists(fname) for fname in (main_fault_file, name_file)])
        fault_names = pd.read_csv(name_file, header=None, squeeze=True, sep='\s+', usecols=[0])

        if from_pickle:
            all_fault_df = pd.read_pickle(main_fault_file)
        else:
            # Prepare info (types and headers) about columns
            column_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                            "slip_rate", "fault_num", "bruce_name"]

            # Read in data
            all_fault_df = pd.read_csv(main_fault_file, sep='\s+', header=None, comment='#', names=column_names, usecols=range(len(column_names)))

        fault_numbers = all_fault_df.fault_num.to_numpy()
        fault_names_unique = dict.fromkeys(fault_names).keys()
        fault_num_unique = dict.fromkeys(fault_numbers).keys()

        assert len(fault_names_unique) == len(fault_num_unique)

        # Populate faults with triangular patches
        patch_start = 0
        segment_ls = []

        for fault_num, fault_name in zip(fault_num_unique, fault_names_unique):
            mask = fault_numbers == fault_num
            fault_data = all_fault_df[mask]
            fault_name_stripped = fault_name.lstrip("'[")

            num_triangles = len(fault_data)
            patch_numbers = np.arange(patch_start, patch_start + num_triangles)

            if from_pickle:
                fault_i = RsqSimSegment.from_pickle(fault_data, fault_num, patch_numbers, fault_name_stripped)
            else:
                fault_i = RsqSimSegment.from_pandas(fault_data, fault_num, patch_numbers, fault_name_stripped,
                                                    transform_from_utm=transform_from_utm)

            segment_ls.append(fault_i)
            patch_start += num_triangles

        multi_fault = cls(segment_ls)

        return multi_fault

    def pickle_model(self, file: str):
        column_names = ["vertices", "normal_vector", "down_dip_vector", "dip", "along_strike_vector",
                        "centre", "area", "dip_slip", "strike_slip", "fault_num"]

        patches = self.patch_dic.values()
        patch_data = []
        for patch in patches:
            patch_data.append([patch.vertices, patch.normal_vector, patch.down_dip_vector, patch.dip,
                               patch.along_strike_vector, patch.centre, patch.area, patch.dip_slip, patch.strike_slip,
                               patch.segment.segment_number])
        df = pd.DataFrame(patch_data, columns=column_names)
        df.to_pickle(file)

    @classmethod
    def read_cfm_directory(cls, directory: str = None, files: Union[str, list, tuple] = None, shapefile=None):
        # TODO: finish writing
        directory_vs_file = [directory is None, files is None]
        assert not all(directory_vs_file)
        assert any(directory_vs_file)

        # Turn input of whatever format into list of ts files to read.
        if directory is not None:
            assert isinstance(directory, str)
            assert os.path.exists(directory)
            if directory[-1] != "/":
                ts_dir = directory + "/"
            else:
                ts_dir = directory

            ts_list = glob.glob(ts_dir + "*.ts")

        else:
            if isinstance(files, str):
                ts_list = [files]
            else:
                assert all([isinstance(fname, str) for fname in files])
                ts_list = list(files)

        assert len(ts_list) > 0, "No .ts files found..."

        for ts_file in ts_list:
            # Get fault name for comparison with shapefile
            ts_no_path = os.path.basename(ts_file)
            ts_name = ts_no_path.split(".ts")[0]

    def plot_faults_2d(self, fault_list: Iterable = None, show: bool = True, write: str = None):
        if fault_list is not None:
            assert isinstance(fault_list, Iterable)
            assert any([fault.lower() in self.names for fault in fault_list])
            valid_names = []
            for fault_name in fault_list:
                if fault_name not in self.names:
                    print("Fault not found: {}".format(fault_name))
                else:
                    valid_names.append(fault_name)
            assert valid_names, "No valid fault names supplied"
        else:
            valid_names = self.names

        # Find boundary
        valid_faults = [self.name_dic[name] for name in valid_names]
        x1 = min([min(fault.vertices[:, 0]) for fault in valid_faults])
        y1 = min([min(fault.vertices[:, 1]) for fault in valid_faults])
        x2 = max([max(fault.vertices[:, 0]) for fault in valid_faults])
        y2 = max([max(fault.vertices[:, 1]) for fault in valid_faults])
        boundary = [x1, y1, x2, y2]

        fig, ax = plt.subplots()
        for name in valid_names:
            fault_i = self.name_dic[name]
            fault_i.plot_2d(ax)

        plot_coast(ax, clip_boundary=boundary)
        ax.set_aspect("equal")
        if write is not None:
            fig.savefig(write, dpi=300)
        if show:
            fig.show()

    def search_name(self, search_string: str):
        """
        Search fault names using wildcard string
        """
        assert isinstance(search_string, str)
        return [name for name in self.names if fnmatch.fnmatch(name, search_string.lower())]

    def find_closest_patches(self, x, y):
        """
        Finds the closest patches to specified coordinates.
        """
        sq_dist = [(fault, vertex, (x-vertex[0])**2 + (y-vertex[1])**2) for fault in self.faults for vertex in fault.vertices]
        closest_fault, closest_point, min_dist = min(sq_dist, key = lambda t: t[2])
        patches = [patch.patch_number for patch in closest_fault.patch_outlines if np.equal(closest_point, patch.vertices).all(axis=1).any()]
        return patches


def read_bruce(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce/rundir4627",
               fault_file: str = "zfault_Deepen.in", names_file: str = "znames_Deepen.in"):
    fault_full = os.path.join(run_dir, fault_file)
    names_full = os.path.join(run_dir, names_file)

    bruce_faults = RsqSimMultiFault.read_fault_file_bruce(fault_full,
                                                          names_full,
                                                          transform_from_utm=True)
    return bruce_faults
