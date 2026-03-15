"""
RSQSim multi-fault model container and related utilities.

Provides :func:`check_unique_vertices` for duplicate-vertex detection
and :class:`RsqSimMultiFault` for managing a collection of
:class:`~rsqsim_api.fault.segment.RsqSimSegment` objects, including
I/O from RSQSim/Bruce fault files, CFM tsurf directories, and various
GIS export methods.
"""
import glob
import os
from collections.abc import Iterable
from typing import Union
import fnmatch
import pickle

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

from shapely.ops import linemerge
from shapely.geometry import LineString,MultiLineString

def check_unique_vertices(vertex_array: np.ndarray, tolerance: Union[int, float] = 1):
    """
    Check whether any vertices in a 3-D array are potential duplicates.

    Sorts the vertex array and computes distances between adjacent
    rows in the sorted order; vertices closer than ``tolerance`` are
    flagged as potential duplicates.

    Parameters
    ----------
    vertex_array : numpy.ndarray of shape (n, 3)
        3-D vertex coordinates.
    tolerance : int or float, optional
        Distance threshold (m) below which vertices are reported as
        potential duplicates.  Defaults to 1.

    Returns
    -------
    num_closer : int
        Number of adjacent-sorted-vertex pairs closer than
        ``tolerance``.
    tolerance : float
        The tolerance value used.
    """
    assert isinstance(vertex_array, np.ndarray)

    assert vertex_array.shape[1] == 3
    sorted_vertices = np.sort(vertex_array, axis=0)
    differences = sorted_vertices[1:] - sorted_vertices
    distances = np.linalg.norm(differences, axis=1)

    num_closer = len(distances[np.where(distances <= tolerance)])

    return num_closer, float(tolerance)



class RsqSimMultiFault:
    """
    Container for a collection of fault segments forming a complete fault model.

    Aggregates :class:`~rsqsim_api.fault.segment.RsqSimSegment`
    objects, provides name-based lookup, boundary computation, and
    I/O methods for RSQSim/Bruce fault files, CFM tsurf directories,
    and various GIS export formats.

    Attributes
    ----------
    faults : list
        List of :class:`~rsqsim_api.fault.segment.RsqSimSegment`
        (or nested :class:`RsqSimMultiFault`) objects.
    patch_dic : dict
        Mapping of global patch number to patch object.
    faults_with_patches : dict
        Mapping of global patch number to the owning segment.
    """

    def __init__(self, faults: Union[list, tuple, set], crs: int = 2193):
        """
        Parameters
        ----------
        faults : list, tuple, or set
            Fault segment objects.
        crs : int, optional
            EPSG code for the coordinate reference system.
            Defaults to 2193 (NZTM).
        """
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
        Return the fault(s) associated with a given patch number or list of patch numbers.

        Parameters
        ----------
        patch_ls : int or array-like of int
            Patch number(s) to look up.
        fault_from_single_patch : bool, optional
            If ``True`` and a single patch number is given, return the
            owning :class:`~rsqsim_api.fault.segment.RsqSimSegment`
            rather than the patch object.  Defaults to ``False``.

        Returns
        -------
        RsqSimTriangularPatch or RsqSimSegment or RsqSimMultiFault
            The patch object, segment, or a new
            :class:`RsqSimMultiFault` containing the relevant segments.
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
        """Populate :attr:`names` and :attr:`name_dic` from the loaded faults."""
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

    def get_patch_areas(self):
        """
        Return a flat array of patch areas for all faults.

        Returns
        -------
        numpy.ndarray of shape (n_total_patches,)
            Patch areas in m².
        """
        return np.hstack([fault.patch_areas for fault in self.faults])

    def make_v2_name_dic(self,path2cfm: str):
        """
        Build a dictionary mapping Bruce-V2 fault names to CFM fault names.

        Uses fuzzy string matching to find the closest name in the CFM
        fault model GIS file, with hardcoded overrides for ambiguous
        cases (Wairau, Hikurangi, Puysegur).

        Parameters
        ----------
        path2cfm : str
            Path to the CFM fault model GIS file (shapefile or
            GeoJSON) containing a ``"Name"`` column.

        Returns
        -------
        None
            Populates :attr:`v2_name_dic`.

        Raises
        ------
        AssertionError
            If ``path2cfm`` does not exist.
        """
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
        Read an RSQSim fault file in Keith Richards-Dinger's convention.

        The file has 13 whitespace-separated columns:
        ``x1,y1,z1, x2,y2,z2, x3,y3,z3, rake, slip_rate, fault_num, fault_name``.

        Parameters
        ----------
        fault_file : str
            Path to the RSQSim fault input file.
        verbose : bool, optional
            If ``True``, print per-fault progress and duplicate-vertex
            warnings.  Defaults to ``False``.
        crs : int, optional
            EPSG code for the coordinate reference system.
            Defaults to 2193 (NZTM).
        read_slip_rate : bool, optional
            If ``True`` (default), populate patch slip rates.

        Returns
        -------
        RsqSimMultiFault
        """
        assert os.path.exists(fault_file)

        # Prepare info (types and headers) about columns
        column_dtypes = [float] * 11 + [int] + ["U50"]
        column_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                        "slip_rate", "fault_num", "fault_name"]

        # Read in data
        try:
            data = np.genfromtxt(fault_file, dtype=column_dtypes, names=column_names).T
        except ValueError:
            data = np.genfromtxt(fault_file, dtype=column_dtypes[:-1], names=column_names[:-1]).T

        all_fault_nums = np.unique(data["fault_num"])
        num_faults = len(all_fault_nums)
        if len(data.dtype.names) == 13:
            all_fault_names = np.unique(data["fault_name"])
        else:
            all_fault_names = []

        if not len(all_fault_names) == num_faults:
            print("Warning: not every fault has a corresponding name")

        # Populate faults with triangular patches
        patch_start = 0
        segment_ls = []

        for number in all_fault_nums:
            fault_data = data[data["fault_num"] == number]
            # Check that fault number has only one name associated with it
            if "fault_name" in data.dtype.names:
                associated_names = np.unique(fault_data["fault_name"])
            else:
                associated_names = np.array([])
            associated_rakes = np.unique(fault_data["rake"])
            associated_slip_rates = np.array(fault_data["slip_rate"])

            fault_name = associated_names[0] if associated_names.size > 0 else None
            fault_rake = fault_data["rake"] if associated_rakes.size > 0 else None

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
        """
        Read an RSQSim fault model in Bruce Shaw's format.

        Reads a two-file layout: a main CSV fault file and a companion
        names file.  Handles the case where multiple fault segments
        share the same name by appending a numeric suffix.

        Parameters
        ----------
        main_fault_file : str
            Path to the main fault geometry file (space-separated,
            13 columns: x1–z3, rake, slip_rate, fault_num, name).
            Can also be a pickled DataFrame if ``from_pickle`` is
            ``True``.
        name_file : str
            Path to the text file listing fault names (one per line,
            in the same order as fault numbers).
        transform_from_utm : bool, optional
            If ``True``, transform vertex coordinates from UTM zone
            59S to NZTM.  Defaults to ``False``.
        from_pickle : bool, optional
            If ``True``, read the main file as a pickled DataFrame.
            Defaults to ``False``.
        crs : int, optional
            EPSG code.  Defaults to 2193 (NZTM).
        read_slip_rate : bool, optional
            If ``True`` (default), populate patch slip rates.

        Returns
        -------
        RsqSimMultiFault
        """
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
                assert read_slip_rate, "from pickle may not read slip rates correctly"
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
        """
        Serialise all patch data to a pickled pandas DataFrame.

        Parameters
        ----------
        file : str
            Output pickle file path.
        """
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
        """
        Plot selected fault segments in map view coloured by slip rate.

        Parameters
        ----------
        fault_list : iterable of str or None, optional
            Names of faults to plot.  Defaults to all faults.
        show : bool, optional
            If ``True``, call ``fig.show()``.  Defaults to ``False``.
        write : str or None, optional
            Output file path for saving the figure.
        """
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
        """
        Plot fault surface traces on a map axes.

        Parameters
        ----------
        fault_list : iterable of str or None, optional
            Names of faults to plot.  Defaults to all faults.
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw onto.  A new figure is created if ``None``.
        edgecolor : str, optional
            Trace edge colour.  Defaults to ``"r"``.
        linewidth : int or float, optional
            Line width.  Defaults to 0.1.
        clip_bounds : list or None, optional
            ``[x1, y1, x2, y2]`` bounding box for the coastline.
        linestyle : str, optional
            Line style.  Defaults to ``"-"``.
        facecolor : str, optional
            Not currently used.  Defaults to ``"0.8"``.

        Returns
        -------
        matplotlib.axes.Axes
        """
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
        """
        Write fault surface traces to shapefiles.

        Parameters
        ----------
        fault_list : iterable of str or None, optional
            Names of faults to export.  Defaults to all faults.
        prefix : str, optional
            Output file prefix.  Two shapefiles are written:
            ``{prefix}.shp`` and ``{prefix}_traces.shp``.
            Defaults to ``"./bruce_faults"``.
        crs : str, optional
            CRS string for the output shapefile.
            Defaults to ``"EPSG:2193"``.
        """
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
        """
        Write fault patch-outline polygons to a shapefile.

        Parameters
        ----------
        fault_list : iterable of str or None, optional
            Names of faults to export.  Defaults to all faults.
        prefix : str, optional
            Output file prefix; produces ``{prefix}_outlines.shp``.
            Defaults to ``"./bruce_faults"``.
        crs : str, optional
            CRS string.  Defaults to ``"EPSG:2193"``.
        """
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
                        min_slip_rate: float = None, nztm_to_lonlat: bool = False,
                        mm_per_year: bool = True):
        """
        Build a flat array of triangle vertex coordinates plus slip rate and rake.

        Parameters
        ----------
        include_zeros : bool, optional
            If ``True`` (default) and ``min_slip_rate`` is set,
            include below-threshold patches with slip and rake set to
            zero.
        min_slip_rate : float or None, optional
            Minimum slip rate (mm/yr if ``mm_per_year`` is ``True``)
            for inclusion.  Defaults to ``None`` (include all).
        nztm_to_lonlat : bool, optional
            If ``True``, output vertex coordinates in WGS84
            (longitude, latitude).  Defaults to ``False``.
        mm_per_year : bool, optional
            If ``True`` (default), convert slip rates from m/s to
            mm/yr.

        Returns
        -------
        numpy.ndarray of shape (n_patches, 11)
            Columns: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3, slip_rate, rake]``.
        """
        all_patches = []

        for fault in self.faults:
            for patch_id in fault.patch_numbers:
                patch = fault.patch_dic[patch_id]
                if nztm_to_lonlat:
                    triangle_corners = patch.vertices_lonlat.flatten()
                else:
                    triangle_corners = patch.vertices.flatten()
                slip_rate = patch.total_slip
                if mm_per_year:
                    slip_rate = slip_rate * csts.seconds_per_year * 1000.
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
                          min_slip_rate: float = None, nztm_to_lonlat: bool = False, mm_per_year: bool = True):
        """
        Build a :class:`meshio.Mesh` containing slip rate and rake as cell data.

        Parameters
        ----------
        include_zeros : bool, optional
            Include below-threshold patches as zeros.  Defaults to
            ``True``.
        min_slip_rate : float or None, optional
            Minimum slip rate threshold.  Defaults to ``None``.
        nztm_to_lonlat : bool, optional
            Output in WGS84.  Defaults to ``False``.
        mm_per_year : bool, optional
            Convert to mm/yr.  Defaults to ``True``.

        Returns
        -------
        meshio.Mesh
        """
        slip_rate_array = self.slip_rate_array(include_zeros=include_zeros,
                                               min_slip_rate=min_slip_rate, nztm_to_lonlat=nztm_to_lonlat,
                                               mm_per_year=mm_per_year)

        mesh = array_to_mesh(slip_rate_array[:, :9])
        data_dic = {}
        for label, index in zip(["slip", "rake"], [9, 10]):
            data_dic[label] = slip_rate_array[:, index]
        mesh.cell_data = data_dic

        return mesh

    def slip_rate_to_vtk(self, vtk_file: str, include_zeros: bool = True,
                         min_slip_rate: float = None, nztm_to_lonlat: bool = False,
                         mm_per_year: bool = True):
        """
        Write slip rate and rake data to a VTK file.

        Parameters
        ----------
        vtk_file : str
            Output VTK file path.
        include_zeros : bool, optional
            Include below-threshold patches.  Defaults to ``True``.
        min_slip_rate : float or None, optional
            Minimum slip rate threshold.  Defaults to ``None``.
        nztm_to_lonlat : bool, optional
            Output in WGS84.  Defaults to ``False``.
        mm_per_year : bool, optional
            Convert to mm/yr.  Defaults to ``True``.
        """
        mesh = self.slip_rate_to_mesh(include_zeros=include_zeros,
                                      min_slip_rate=min_slip_rate, nztm_to_lonlat=nztm_to_lonlat,
                                      mm_per_year=mm_per_year)
        mesh.write(vtk_file, file_format="vtk")

    def write_rsqsim_input_file(self, output_file: str, mm_yr: bool = True):
        """
        Write all faults to a single RSQSim fault input file.

        Parameters
        ----------
        output_file : str
            Output file path.
        mm_yr : bool, optional
            If ``True`` (default), treat stored slip rates as mm/yr
            and convert to m/s for the output.
        """
        combined_array = pd.concat([fault.to_rsqsim_fault_array(mm_yr=mm_yr) for fault in self.faults])
        combined_array.to_csv(output_file, sep=" ", header=False, index=False)

    def write_b_value_file(self, a_value: float, default_a_b: float, difference_dict: dict, output_file: str):
        """
        Write per-patch b-values to a text file for use with RSQSim.

        Parameters
        ----------
        a_value : float
            Global a-value parameter (0–1).
        default_a_b : float
            Default a − b value (negative float).
        difference_dict : dict
            Mapping of fault name to (a − b) value override.
        output_file : str
            Output text file path.
        """
        assert isinstance(difference_dict, dict)
        assert all([name in self.names for name in difference_dict.keys()])
        assert all([isinstance(value, float) for value in difference_dict.values()])
        assert isinstance(a_value, float)
        assert a_value > 0.
        assert a_value < 1.
        assert isinstance(default_a_b, float)
        assert default_a_b < 0.
        assert isinstance(output_file, str)
        combined_array = pd.concat([fault.to_rsqsim_fault_array() for fault in self.faults])
        default_b = a_value - default_a_b
        default_b_array = np.ones(len(combined_array)) * default_b
        for fault_name, difference in difference_dict.items():
            fault_index = np.where(combined_array.iloc[:, 12] == fault_name)
            default_b_array[fault_index] = a_value - difference
        np.savetxt(output_file, default_b_array.T, fmt="%.5f")

    def tile_quads(self, tile_size: float = 5000., interpolation_distance: float = 1000.,
                         manual_tiles: dict = None, output_file: str = None):
        """
        Discretize all faults into rectangular tiles.

        Calls :meth:`~rsqsim_api.fault.segment.RsqSimSegment.discretize_rectangular_tiles`
        on each fault.  Manual tile arrays can override individual
        faults.

        Parameters
        ----------
        tile_size : float, optional
            Target tile dimension (m).  Defaults to 5000.
        interpolation_distance : float, optional
            Down-dip interpolation spacing (m).  Defaults to 1000.
        manual_tiles : dict or None, optional
            Mapping of fault name to pre-computed tile array (loaded
            from a ``.npy`` file).  Defaults to ``None``.
        output_file : str or None, optional
            If given, pickle the resulting dict to this file.

        Returns
        -------
        dict
            Mapping of fault name to tile array (shape (n_tiles, 4, 3)).
        """
        if output_file is not None:
            assert isinstance(output_file, str)
        assert isinstance(tile_size, float)
        assert isinstance(interpolation_distance, float)
        assert isinstance(manual_tiles, dict) or manual_tiles is None

        quads_dict = {}
        if manual_tiles is not None:
            assert all([key in self.names for key in manual_tiles.keys()])
            for fault_name, file_name in manual_tiles.items():
                assert os.path.exists(file_name)
                tiles_array = np.load(file_name)
                quads_dict[fault_name] = tiles_array


        for fault in self.faults:
            if not fault.name in manual_tiles.keys():
                try:
                    quads = fault.discretize_rectangular_tiles(tile_size=tile_size,
                                                               interpolation_distance=interpolation_distance)
                    quads = np.array(quads)

                except Exception as e:
                    print("Failed to merge", fault.name)
                    print(e)
                    quads = np.array([])

                quads_dict[fault.name] = quads
        if output_file is not None:
            with open(output_file, "wb") as f:
                pickle.dump(quads_dict, f)

        return quads_dict







    def search_name(self, search_string: str):
        """
        Search fault names using a wildcard pattern.

        Parameters
        ----------
        search_string : str
            Wildcard pattern (case-insensitive) to match against
            fault names.

        Returns
        -------
        list of str
            Fault names that match the pattern.
        """
        assert isinstance(search_string, str)
        return [name for name in self.names if fnmatch.fnmatch(name, search_string.lower())]

    def find_closest_patches(self, x, y):
        """
        Find the patches closest to given 2-D coordinates.

        Parameters
        ----------
        x : float
            Easting (NZTM) in metres.
        y : float
            Northing (NZTM) in metres.

        Returns
        -------
        list of int
            Patch numbers of the patches in the closest fault whose
            vertices include the nearest vertex.
        """
        sq_dist = [(fault, vertex, (x - vertex[0]) ** 2 + (y - vertex[1]) ** 2) for fault in self.faults for vertex in
                   fault.vertices]
        closest_fault, closest_point, min_dist = min(sq_dist, key=lambda t: t[2])
        patches = [patch.patch_number for patch in closest_fault.patch_outlines if
                   np.equal(closest_point, patch.vertices).all(axis=1).any()]
        return patches

    def find_closest_patches_3d(self, x, y, z):
        """
        Find the patches closest to given 3-D coordinates.

        Parameters
        ----------
        x : float
            Easting (NZTM) in metres.
        y : float
            Northing (NZTM) in metres.
        z : float
            Depth (negative metres below sea level).

        Returns
        -------
        list of int
            Patch numbers of the patches in the closest fault whose
            vertices include the nearest vertex.
        """
        sq_dist = [(fault, vertex, (x - vertex[0]) ** 2 + (y - vertex[1]) ** 2 + (z - vertex[2]) ** 2 ) for fault in self.faults for vertex in
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
        """Build and cache a GeoDataFrame of all fault surface traces."""
        traces = [fault.trace for fault in self.faults]
        trace_gpd = gpd.GeoDataFrame({"fault": self.names}, geometry=traces, crs=self.crs)
        self._traces = trace_gpd

    @property
    def outlines(self):
        if self._outlines is None:
            self.get_outlines()
        return self._outlines

    def get_outlines(self):
        """Build and cache a GeoDataFrame of all fault patch-outline polygons."""
        outlines = [fault.fault_outline for fault in self.faults]
        outline_gpd = gpd.GeoDataFrame({"fault": self.names}, geometry=outlines, crs=self.crs)
        self._outlines = outline_gpd

    def merge_segments(self, matching_string: str, name_dict: dict = None, fault_name: str = None):
        """
        Merge multiple fault segments matching a name pattern into one segment.

        Parameters
        ----------
        matching_string : str
            Wildcard pattern used to identify segments to merge.
            Without ``name_dict``, matches names like
            ``"{matching_string}?"`` or ``"{matching_string}??"``.
        name_dict : dict or None, optional
            Mapping of internal segment name to an alternative name
            for pattern matching.
        fault_name : str or None, optional
            Name for the merged segment.

        Returns
        -------
        RsqSimSegment
            New segment containing all patches from the matched
            segments, with merged surface trace.
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

    def get_subduction_patch_numbers(self, subduction_faults: tuple[str] = ("hikkerm", "puysegur")):
        """
        Return patch numbers for all subduction-zone fault segments.

        Parameters
        ----------
        subduction_faults : tuple of str, optional
            Substrings used to identify subduction faults by name.
            Defaults to ``("hikkerm", "puysegur")``.

        Returns
        -------
        numpy.ndarray of int
        """
        subduction_patches = []
        for fault in self.faults:
            if any([subduction_fault in fault.name for subduction_fault in subduction_faults]):
                subduction_patches.extend(fault.patch_numbers)
        return np.array(subduction_patches)

    def get_crustal_patch_numbers(self, subduction_faults: tuple[str] = ("hikkerm", "puysegur")):
        """
        Return patch numbers for all crustal (non-subduction) fault segments.

        Parameters
        ----------
        subduction_faults : tuple of str, optional
            Substrings identifying subduction faults to exclude.
            Defaults to ``("hikkerm", "puysegur")``.

        Returns
        -------
        numpy.ndarray of int
        """
        subduction_patches = self.get_subduction_patch_numbers(subduction_faults=subduction_faults)
        all_patches = np.array(list(self.patch_dic.keys()))
        crustal_patches = np.setdiff1d(all_patches, subduction_patches)
        return crustal_patches




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
