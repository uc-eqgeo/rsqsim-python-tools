import glob
import os
from collections.abc import Iterable
from typing import Union
import fnmatch

import numpy as np
import pandas as pd
import geopandas as gpd
import difflib
from matplotlib import pyplot as plt

from rsqsim_api.visualisation.utilities import plot_coast
from rsqsim_api.fault.segment import RsqSimSegment
from rsqsim_api.io.mesh_utils import array_to_mesh
import rsqsim_api.io.rsqsim_constants as csts
from rsqsim_api.fault.utilities import merge_multiple_nearly_adjacent_segments

from shapely.ops import linemerge,split
from shapely.geometry import LineString,MultiLineString

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
    def __init__(self, faults: Union[list, tuple, set], crs: int = 2193):
        self._faults = None
        self.faults = faults
        self._names = None
        self._name_dic = None
        self._crs = crs
        self._traces = None
        self._outlines = None
        self._v2_name_dic = None
        self.patch_dic = {}
        for fault in self.faults:
            if self.patch_dic is not None:
                self.patch_dic.update(fault.patch_dic)

        self.faults_with_patches = {patch_num: patch.segment for patch_num, patch in self.patch_dic.items()}

    def filter_faults_by_patch_numbers(self, patch_ls: Union[int, list, tuple, np.ndarray],fault_from_single_patch : bool =False):
        """

        """
        if isinstance(patch_ls, np.integer):
            if fault_from_single_patch:
                return self.faults_with_patches[patch_ls]
            else:
                return self.patch_dic[patch_ls]
        else:
            assert isinstance(patch_ls, (tuple, list, np.ndarray))
            assert all([isinstance(x, (np.integer, int)) for x in patch_ls])
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

    @property
    def crs(self):
        return self._crs

    @property
    def v2_name_dic(self):
        return self._v2_name_dic

    def make_v2_name_dic(self,path2cfm: str):
        """Make dictionary to convert Bruce V2 fault segment names to the equivalent CFM names"""
        assert os.path.exists(path2cfm), "Path to CFM fault model not found"
        cfm = gpd.read_file(path2cfm)
        cfm_names=[name.lower() for name in cfm['Name']]
        nearest_cfm = [difflib.get_close_matches(name[:-1], cfm_names, n=1) for name in self.names]
        name_dict=dict(zip(self.names,nearest_cfm))
        for key in name_dict.keys():
            if not bool(name_dict[key]):
                name_dict[key]=key[:-1].replace(" ","")
            else:
                name_dict[key]=name_dict[key][0].replace(" ","")
        #add in awkward faults
        name_dict['wairau20'] = 'wairau'
        name_dict['wairau30'] = 'wairau'
        # add hikurangi and puysegur
        puy_names = [name for name in self.names if fnmatch.fnmatch(name, "*puysegar*")]
        hikurangi_names = [name for name in self.names if fnmatch.fnmatch(name, "*hikurangi*")]
        for name in puy_names:
            name_dict[name] = 'puy'
        for name in hikurangi_names:
            name_dict[name] = 'hik'

        self._v2_name_dic=name_dict

        return

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
    def read_fault_file_keith(cls, fault_file: str, verbose: bool = False, crs: int = 2193,
                              read_slip_rate: bool = True):
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
            associated_rakes = np.unique(fault_data["rake"])
            associated_slip_rates = np.array(fault_data["slip_rate"])

            fault_name = associated_names[0] if associated_names.size > 0 else None
            fault_rake = associated_rakes[0] if associated_rakes.size > 0 else None

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
            if read_slip_rate:
                fault_i = RsqSimSegment.from_triangles(triangles=triangles, patch_numbers=patch_numbers,
                                                       segment_number=number, fault_name=fault_name, rake=fault_rake,
                                                       total_slip=associated_slip_rates)
            else:
                fault_i = RsqSimSegment.from_triangles(triangles=triangles, patch_numbers=patch_numbers,
                                                       segment_number=number, fault_name=fault_name, rake=fault_rake)

            segment_ls.append(fault_i)
            patch_start += num_triangles

        multi_fault = cls(segment_ls, crs=crs)

        return multi_fault

    @classmethod
    def read_fault_file_bruce(cls, main_fault_file: str, name_file: str, transform_from_utm: bool = False,
                              from_pickle: bool = False, crs: int = 2193, read_slip_rate: bool = True):
        assert all([os.path.exists(fname) for fname in (main_fault_file, name_file)])
        with open(name_file) as fid:
            names_strings = fid.readlines()
            if names_strings[0].strip()[:3] == "[b'":
                fault_names = [name.strip()[3:-2].strip().split(",")[0] for name in names_strings]
            elif names_strings[0].strip()[:2] == "['":
                fault_names = [name.strip()[2:-4].strip() for name in names_strings]
            else:
                fault_names = [name.strip() for name in names_strings]

        if from_pickle:
            all_fault_df = pd.read_pickle(main_fault_file)
        else:
            # Prepare info (types and headers) about columns
            column_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                            "slip_rate", "fault_num", "bruce_name"]

            # Read in data
            all_fault_df = pd.read_csv(main_fault_file, sep='\s+', header=None, comment='#', names=column_names,
                                       usecols=range(len(column_names)))

        all_fault_df["fault_name"] = fault_names

        fault_numbers = all_fault_df.fault_num.to_numpy()
        fault_names_unique = dict.fromkeys(fault_names).keys()
        fault_num_unique = dict.fromkeys(fault_numbers).keys()

        if len(fault_names_unique) != len(fault_num_unique):
            names_and_numbers = pd.DataFrame({"Name": fault_names, "Num": fault_numbers})
            for name in fault_names_unique:
                name_num_i = names_and_numbers[names_and_numbers.Name == name]
                for i, fault_num in enumerate(name_num_i.Num.unique()):
                    fault_num_df = name_num_i[name_num_i.Num == fault_num]
                    new_names = fault_num_df.Name.astype(str) + " " + str(i)
                    names_and_numbers.loc[new_names.index, "Name"] = new_names.values

            acton = names_and_numbers.loc[
                (names_and_numbers.Name.str.contains("Acton")) & (names_and_numbers.Num == 120), "Num"]
            all_fault_df.loc[acton.index, "fault_num"] = 9999
            names_and_numbers.loc[acton.index, "Num"] = 9999

            fault_names = list(names_and_numbers.Name)
            fault_names_unique = dict.fromkeys(fault_names).keys()
            fault_num_unique = dict.fromkeys(list(names_and_numbers.Num)).keys()
        assert len(fault_names_unique) == len(fault_num_unique)

        # Populate faults with triangular patches
        patch_start = 0
        segment_ls = []

        for fault_num, fault_name in zip(fault_num_unique, fault_names_unique):
            mask = fault_numbers == fault_num
            fault_data = all_fault_df[mask]
            fault_name_stripped = fault_name.lstrip("'[").replace(" ", "")

            num_triangles = len(fault_data)
            patch_numbers = np.arange(patch_start, patch_start + num_triangles)

            if from_pickle:
                assert read_slip_rate, "from pickle may not read sip rates correctly"
                fault_i = RsqSimSegment.from_pickle(fault_data, fault_num, patch_numbers, fault_name_stripped)
            else:
                fault_i = RsqSimSegment.from_pandas(fault_data, fault_num, patch_numbers, fault_name_stripped,
                                                        transform_from_utm=transform_from_utm,
                                                        read_slip_rate=read_slip_rate)

            segment_ls.append(fault_i)
            patch_start += num_triangles

        multi_fault = cls(segment_ls, crs=crs)

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

    def plot_faults_2d(self, fault_list: Iterable = None, show: bool = False, write: str = None):
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

    def plot_fault_traces(self, fault_list: Iterable = None, ax: plt.Axes = None, edgecolor: str = "r", linewidth: int = 0.1, clip_bounds: list = None,
                            linestyle: str = "-", facecolor: str = "0.8"):
        # TODO: fix projection issue which means fault traces aren't plotted on correct scale
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
        x1 = min([min(fault.vertices[:, 0]) for fault in valid_faults])*0.99
        y1 = min([min(fault.vertices[:, 1]) for fault in valid_faults])*0.99
        x2 = max([max(fault.vertices[:, 0]) for fault in valid_faults])*1.01
        y2 =max([max(fault.vertices[:, 1]) for fault in valid_faults])*1.01

        boundary = [x1, y1, x2, y2]
        #print(boundary)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            assert isinstance(ax,plt.Axes)
        # plot traces
        for fault in valid_faults:
            faultArr=np.array(fault.trace.coords)
            ax.plot(faultArr[:,0],faultArr[:,1],"r")
            print(np.array(fault.trace.coords))
        x1,y1,x2,y2=plot_coast(ax, clip_boundary=boundary)

        ax.set_xlim(x1,x2)
        ax.set_ylim(y1,y2)
        ax.set_aspect("equal")

        return ax

    def write_fault_traces_to_gis(self, fault_list: Iterable = None, prefix: str = "./bruce_faults",crs: str ="EPSG:2193" ):
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
        valid_faults = [self.name_dic[name] for name in valid_names]
        fault_dict = {'Names': [], 'geometry': [], 'Slip Rate': []}
        for fault in valid_faults:
            trace = fault.trace
            fName = fault.name

            mean_slip_rate = fault.mean_slip_rate * csts.seconds_per_year * 1000.
            fault_dict['Names'].append(fName)
            fault_dict['geometry'].append(trace)
            fault_dict['Slip Rate'].append(mean_slip_rate)

        all_faults = gpd.GeoDataFrame.from_dict(fault_dict)
        all_faults.to_file(prefix+".shp", crs=crs)
        all_faults.to_file(prefix+"_traces.shp", crs=crs)

    def write_fault_outlines_to_gis(self, fault_list: Iterable = None, prefix: str = "./bruce_faults",crs: str ="EPSG:2193" ):
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
        valid_faults = [self.name_dic[name] for name in valid_names]
        fault_dict = {'Names': [], 'geometry': [], 'Slip Rate': []}
        for fault in valid_faults:
            outline = fault.fault_outline
            fName = fault.name

            mean_slip_rate = fault.mean_slip_rate * csts.seconds_per_year * 1000.
            fault_dict['Names'].append(fName)
            fault_dict['geometry'].append(outline)
            fault_dict['Slip Rate'].append(mean_slip_rate)

        all_faults = gpd.GeoDataFrame.from_dict(fault_dict)
        all_faults.to_file(prefix+"_outlines.shp", crs=crs)


    def plot_slip_distribution_2d(self):
        pass


    def slip_rate_array(self, include_zeros: bool = True,
                        min_slip_rate: float = None, nztm_to_lonlat: bool = False):
        all_patches = []

        for fault in self.faults:
            for patch_id in fault.patch_numbers:
                patch = fault.patch_dic[patch_id]
                if nztm_to_lonlat:
                    triangle_corners = patch.vertices_lonlat.flatten()
                else:
                    triangle_corners = patch.vertices.flatten()
                slip_rate = patch.total_slip
                if min_slip_rate is not None:
                    if slip_rate >= min_slip_rate:
                        patch_line = np.hstack([triangle_corners, np.array([slip_rate, patch.rake])])
                        all_patches.append(patch_line)
                    elif include_zeros:
                        patch_line = np.hstack([triangle_corners, np.array([0., 0.])])
                        all_patches.append(patch_line)
                else:
                    patch_line = np.hstack([triangle_corners, np.array([slip_rate, patch.rake])])
                    all_patches.append(patch_line)

        return np.array(all_patches)

    def slip_rate_to_mesh(self, include_zeros: bool = True,
                          min_slip_rate: float = None, nztm_to_lonlat: bool = False):

        slip_rate_array = self.slip_rate_array(include_zeros=include_zeros,
                                               min_slip_rate=min_slip_rate, nztm_to_lonlat=nztm_to_lonlat)

        mesh = array_to_mesh(slip_rate_array[:, :9])
        data_dic = {}
        for label, index in zip(["slip", "rake"], [9, 10]):
            data_dic[label] = slip_rate_array[:, index]
        mesh.cell_data = data_dic

        return mesh

    def slip_rate_to_vtk(self, vtk_file: str, include_zeros: bool = True,
                         min_slip_rate: float = None, nztm_to_lonlat: bool = False):
        mesh = self.slip_rate_to_mesh(include_zeros=include_zeros,
                                      min_slip_rate=min_slip_rate, nztm_to_lonlat=nztm_to_lonlat)
        mesh.write(vtk_file, file_format="vtk")

    def write_rsqsim_input_file(self, output_file: str):
        combined_array = pd.concat([fault.to_rsqsim_fault_array() for fault in self.faults])
        combined_array.to_csv(output_file, sep=" ", header=False, index=False)

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
        sq_dist = [(fault, vertex, (x - vertex[0]) ** 2 + (y - vertex[1]) ** 2) for fault in self.faults for vertex in
                   fault.vertices]
        closest_fault, closest_point, min_dist = min(sq_dist, key=lambda t: t[2])
        patches = [patch.patch_number for patch in closest_fault.patch_outlines if
                   np.equal(closest_point, patch.vertices).all(axis=1).any()]
        return patches

    @property
    def traces(self):
        if self._traces is None:
            self.get_traces()
        return self._traces

    def get_traces(self):
        traces = [fault.trace for fault in self.faults]
        trace_gpd = gpd.GeoDataFrame({"fault": self.names}, geometry=traces, crs=self.crs)
        self._traces = trace_gpd

    @property
    def outlines(self):
        if self._outlines is None:
            self.get_outlines()
        return self._outlines

    def get_outlines(self):
        outlines = [fault.fault_outline for fault in self.faults]
        outline_gpd = gpd.GeoDataFrame({"fault": self.names}, geometry=outlines, crs=self.crs)
        self._outlines = outline_gpd

    def merge_segments(self, matching_string: str, name_dict: dict = None, fault_name: str = None):
        """
        Merge segments of a fault.
        """
        if name_dict is not None:
            matching_names = [name for name in self.names if any([fnmatch.fnmatch(name_dict[name], f"{matching_string}")])]
        else:
            matching_names=[name for name in self.names if any([fnmatch.fnmatch(name, f"{matching_string.lower()}?"),
                                                              fnmatch.fnmatch(name, f"{matching_string.lower()}??")])]
        matching_faults = [self.name_dic[name] for name in matching_names]
        vertices = np.vstack([fault.patch_vertices_flat for fault in matching_faults])
        patch_numbers = np.hstack([fault.patch_numbers for fault in matching_faults])

        new_segment = RsqSimSegment.from_triangles(vertices, patch_numbers=patch_numbers, fault_name=fault_name)
        traces = [fault.trace for fault in matching_faults]


        if all([isinstance(trace,LineString) for trace in traces]):
            merged_traces=linemerge(traces)
        else:
            trace_list = [list(trace.geoms) for trace in traces if isinstance(trace,MultiLineString)]
            merged_traces=merge_multiple_nearly_adjacent_segments(trace_list[0])
            #print(trace_list)
        #merged_traces = merge_multiple_nearly_adjacent_segments(traces)

        if isinstance(merged_traces,LineString):
            new_segment.trace = merged_traces
        else:
            try:
                merged_coords=[list(geom.coords) for geom in merged_traces.geoms]
                merged_trace=LineString([trace for sublist in merged_coords for trace in sublist])
                new_segment.trace=merged_trace
            except:
                print(f'Check trace type for {fault_name}')

        return new_segment




def read_bruce(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/shaw2021/rundir4627",
               fault_file: str = "bruce_faults.in", names_file: str = "bruce_names.in"):
    fault_full = os.path.join(run_dir, fault_file)
    names_full = os.path.join(run_dir, names_file)

    @property
    def outlines(self):
        if self._outlines is None:
            self.get_outlines()
        return self._outlines

    def get_outlines(self):
        outlines = [fault.fault_outline for fault in self.faults]
        outline_gpd = gpd.GeoDataFrame({"fault": self.names}, geometry=outlines, crs=self.crs)
        self._outlines = outline_gpd
