"""
RSQSim earthquake catalogue management and analysis.

Provides :class:`RsqSimCatalogue` for loading, filtering, and analysing
earthquake catalogues produced by RSQSim, along with module-level helper
functions for reading Bruce Shaw run directories and combining spatial
bounds.

The catalogue stores per-event metadata (time, magnitude, location,
area, duration) as a Pandas DataFrame alongside parallel flat arrays of
per-patch slip, timing, and event identifiers that can be lazily resolved
to :class:`~rsqsim_api.catalogue.event.RsqSimEvent` objects.
"""
from typing import Any
from collections import abc, Counter, defaultdict
from collections.abc import Iterable
import os
import pickle

from multiprocessing import Queue, Process
from multiprocessing import sharedctypes
import pandas as pd
import numpy as np
import pyproj
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from shapely.geometry import Polygon
import geopandas as gpd
from numpy.random import default_rng
from numba import njit, prange, types, typed

from rsqsim_api.fault.multifault import RsqSimMultiFault, RsqSimSegment
from rsqsim_api.catalogue.event import RsqSimEvent
from rsqsim_api.io.read_utils import read_earthquake_catalogue, read_binary, catalogue_columns, read_csv_and_array, read_text
from rsqsim_api.io.write_utils import write_catalogue_dataframe_and_arrays
from rsqsim_api.tsunami.tsunami import SeaSurfaceDisplacements
from rsqsim_api.visualisation.utilities import plot_coast, plot_background, plot_hillshade_niwa, plot_lake_polygons, \
    plot_river_lines, plot_highway_lines, plot_boundary_polygons
from rsqsim_api.io.bruce_shaw_utilities import bruce_subduction
import rsqsim_api.io.rsqsim_constants as csts
from rsqsim_api.catalogue.utilities import calculate_scaling_c, calculate_stress_drop, \
    summary_statistics, median_cumulant, mw_to_m0, jit_intersect

rng = default_rng()

fint = int | float
sensible_ranges = {"t0": (0, 1.e15), "m0": (1.e13, 1.e24), "mw": (2.5, 10.0),
                   "x": (-180., 1.e8), "y": (-90., 1.e8), "z": (-1.e6, 0),
                   "area": (0, 1.e12), "dt": (0, 1200)}

list_file_suffixes = (".pList", ".eList", ".dList", ".tList")
extra_file_suffixes = (".dmuList", ".dsigmaList", ".dtauList", ".taupList")


def get_mask(ev_ls, min_patches, faults_with_patches, event_list, patch_list, queue):
    """
    Worker function: compute per-event patch masks and post results to a queue.

    For each event index in ``ev_ls``, finds the patches belonging to each
    fault segment and masks out segments that contribute fewer than
    ``min_patches`` patches.  Results are posted to ``queue`` as
    ``(event_index, ev_indices, mask)`` tuples.

    Parameters
    ----------
    ev_ls : array-like of int
        Event indices to process in this worker.
    min_patches : int
        Minimum patches per fault for a segment to be retained.
    faults_with_patches : dict
        Mapping of patch number to fault segment number (integer IDs
        rather than objects, for serialisability).
    event_list : array-like of int
        Flat array of event IDs for every patch entry in the catalogue.
    patch_list : array-like of int
        Flat array of patch IDs parallel to ``event_list``.
    queue : multiprocessing.Queue
        Output queue; tuples of ``(event_id, ev_indices, mask)`` are
        placed here.
    """
    patches = np.asarray(patch_list)
    events = np.asarray(event_list)

    unique_events, unique_event_indices = np.unique(events, return_index=True)
    unique_dic = {unique_events[i]: (unique_event_indices[i], unique_event_indices[i + 1]) for i in
                  range(len(unique_events) - 1)}
    unique_dic[unique_events[-1]] = (unique_event_indices[-1], len(events))
    for index in ev_ls:
        ev_range = unique_dic[index]
        ev_indices = np.arange(ev_range[0], ev_range[1])

        patch_numbers = patches[ev_indices]
        patches_on_fault = defaultdict(list)
        [patches_on_fault[faults_with_patches[i]].append(i) for i in patch_numbers]

        mask = np.full(len(patch_numbers), True)
        for fault in patches_on_fault.keys():
            patches_on_this_fault = patches_on_fault[fault]
            if len(patches_on_this_fault) < min_patches:
                patch_on_fault_indices = np.searchsorted(patch_numbers, patches_on_this_fault)
                mask[patch_on_fault_indices] = False

        queue.put((index, ev_indices, mask))


class RsqSimCatalogue:
    """
    Container for an RSQSim earthquake catalogue with per-patch slip data.

    Stores the catalogue as a Pandas DataFrame together with four
    parallel flat arrays (event IDs, patch IDs, slip values, and rupture
    times) that cover the slip distribution for every event.  Provides
    methods to load data from RSQSim binary output files, filter by
    magnitude, time, location, or fault, retrieve fully populated
    :class:`~rsqsim_api.catalogue.event.RsqSimEvent` objects, and
    produce diagnostic plots.

    Attributes
    ----------
    t0, m0, mw : numpy.ndarray or None
        Origin time (s), scalar moment (N·m), and moment magnitude
        arrays from the catalogue DataFrame (set lazily).
    x, y, z : numpy.ndarray or None
        Hypocentre NZTM coordinates (m).
    area, dt : numpy.ndarray or None
        Rupture areas (m²) and durations (s).
    """

    def __init__(self):
        """Initialise an empty catalogue; use class methods to populate."""
        # Essential attributes
        self._catalogue_df = None
        self._event_list = None
        self._patch_list = None
        self._patch_time_list = None
        self._patch_slip = None
        self._accumulated_slip = None
        self._event_mean_slip = None
        self._event_length = None
        self._event_mean_sdr = None
        self._event_length = None
        # Useful attributes
        self.t0, self.m0, self.mw = (None,) * 3
        self.x, self.y, self.z = (None,) * 3
        self.area, self.dt = (None,) * 2

    @property
    def catalogue_df(self):
        """pandas.DataFrame: Per-event catalogue with 8 numeric columns."""
        return self._catalogue_df

    @catalogue_df.setter
    def catalogue_df(self, dataframe: pd.DataFrame):
        assert dataframe.columns.size == 8, "Should have 8 columns"
        assert all([col.dtype in ("float", "int") for i, col in dataframe.items()])
        dataframe.columns = catalogue_columns
        self._catalogue_df = dataframe

    def check_list(self, data_list: np.ndarray, data_type: str):
        """
        Validate a flat patch/event list array before assignment.

        Parameters
        ----------
        data_list : numpy.ndarray
            1-D array to validate.
        data_type : str
            ``"i"`` for integer arrays (patch/event lists) or ``"d"``
            for floating-point arrays (slip/time lists).

        Raises
        ------
        AttributeError
            If the catalogue DataFrame has not been loaded yet.
        AssertionError
            If ``data_type``, dtype, or dimensionality are invalid.
        """
        assert data_type in ("i", "d")
        if self.catalogue_df is None:
            raise AttributeError("Read in main catalogue (eqs.*.out) before list files")
        if data_type == "i":
            assert data_list.dtype.char in np.typecodes['AllInteger']
        else:
            assert data_list.dtype.char in np.typecodes['AllFloat']
        assert data_list.ndim == 1, "Expecting 1D array as input"
        return

    @property
    def event_list(self):
        """numpy.ndarray: Flat integer array of event IDs, one per patch entry."""
        return self._event_list

    @event_list.setter
    def event_list(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="i")
        if not len(np.unique(data_list)) == len(self.catalogue_df):
            print(len(np.unique(data_list)),len(self.catalogue_df))
            raise ValueError("Numbers of events in catalogue and supplied list are different!")
        self._event_list = data_list

    @property
    def patch_list(self):
        """numpy.ndarray: Flat integer array of patch IDs, parallel to ``event_list``."""
        return self._patch_list

    @patch_list.setter
    def patch_list(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="i")
        self._patch_list = data_list

    @property
    def patch_time_list(self):
        """numpy.ndarray: Rupture times (s) for each patch entry."""
        return self._patch_time_list

    @patch_time_list.setter
    def patch_time_list(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="d")
        self._patch_time_list = data_list

    @property
    def patch_slip(self):
        """numpy.ndarray: Slip magnitudes (m) for each patch entry."""
        return self._patch_slip

    @patch_slip.setter
    def patch_slip(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="d")
        self._patch_slip = data_list

    @property
    def accumulated_slip(self):
        """dict: Mapping of patch ID to total slip accumulated over all events."""
        if self._accumulated_slip is None:
            self.assign_accumulated_slip()
        return self._accumulated_slip

    @property
    def event_mean_slip(self):
        """dict or None: Mapping of event ID to mean slip (m); populated by :meth:`assign_event_mean_slip`."""
        return self._event_mean_slip

    @property
    def event_mean_sdr(self):
        """dict or None: Mapping of event ID to ``[strike, dip, rake]``; populated by :meth:`assign_event_mean_sdr`."""
        return self._event_mean_sdr

    @property
    def event_length(self):
        """dict or None: Mapping of event ID to rupture length (m); populated by :meth:`assign_event_length`."""
        return self._event_length

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, reproject: List = None):
        """
        Construct a catalogue from an existing DataFrame.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame with 8 numeric columns in the standard RSQSim
            catalogue column order (t0, m0, mw, x, y, z, area, dt).
        reproject : list or tuple of int, optional
            ``[in_epsg, out_epsg]`` pair.  If provided, the ``x`` and
            ``y`` columns are reprojected using PyProj.

        Returns
        -------
        RsqSimCatalogue
            Catalogue populated with the DataFrame (no patch lists).
        """
        rsqsim_cat = cls()
        if reproject is not None:
            assert isinstance(reproject, (tuple, list))
            assert len(reproject) == 2
            in_epsg, out_epsg = reproject
            transformer = pyproj.Transformer.from_crs(in_epsg, out_epsg, always_xy=True)
            new_x, new_y = transformer.transform(dataframe["x"], dataframe["y"])
            dataframe["x"] = new_x
            dataframe["y"] = new_y
        rsqsim_cat.catalogue_df = dataframe
        return rsqsim_cat

    @classmethod
    def from_catalogue_file(cls, filename: str, reproject: List = None):
        """
        Construct a catalogue by reading an RSQSim ``eqs.*.out`` file.

        Parameters
        ----------
        filename : str
            Path to the RSQSim catalogue file.
        reproject : list or tuple of int, optional
            ``[in_epsg, out_epsg]`` pair for coordinate reprojection.

        Returns
        -------
        RsqSimCatalogue
            Catalogue populated from the file (no patch lists).
        """
        assert os.path.exists(filename)
        catalogue_df = read_earthquake_catalogue(filename)
        rsqsim_cat = cls.from_dataframe(catalogue_df, reproject=reproject)
        return rsqsim_cat

    @classmethod
    def from_catalogue_file_and_lists(cls, catalogue_file: str, list_file_directory: str,
                                      list_file_prefix: str, read_extra_lists: bool = False, reproject: List = None, serial: bool = False, endian: str = "little"):
        """
        Construct a fully populated catalogue from an RSQSim output directory.

        Reads the ``eqs.*.out`` catalogue file together with the four
        binary list files (``.pList``, ``.eList``, ``.dList``, ``.tList``).

        Parameters
        ----------
        catalogue_file : str
            Path to the ``eqs.*.out`` catalogue file.
        list_file_directory : str
            Directory containing the ``.pList``/``.eList``/``.dList``/
            ``.tList`` binary files.
        list_file_prefix : str
            Common prefix for the list files (e.g. ``"rundir4627"``).
        read_extra_lists : bool, optional
            Unused; reserved for future extra list support.
        reproject : list or tuple of int, optional
            ``[in_epsg, out_epsg]`` pair for coordinate reprojection.
        serial : bool, optional
            If ``True``, read list files as plain text rather than
            binary.  Defaults to ``False``.
        endian : str, optional
            Byte order of the binary list files: ``"little"`` (default)
            or ``"big"``.

        Returns
        -------
        RsqSimCatalogue
            Fully populated catalogue with patch slip distribution.
        """
        assert os.path.exists(catalogue_file)
        assert os.path.exists(list_file_directory)

        standard_list_files = [os.path.join(list_file_directory, list_file_prefix + suffix)
                               for suffix in list_file_suffixes]
        for fname, suffix in zip(standard_list_files, list_file_suffixes):
            if not os.path.exists(fname):
                raise FileNotFoundError("{} file required to populate event slip distributions".format(suffix))

        # Read in catalogue to dataframe and initiate class instance
        rcat = cls.from_catalogue_file(catalogue_file, reproject=reproject)

        num_events = len(rcat.catalogue_df)

        if serial:

            event_list = read_text(standard_list_files[1], format="i") - 1
            patch_list = read_text(standard_list_files[0], format="i") - 1
            patch_slip, patch_time_list = [read_text(fname, format="d") for fname in
                                           standard_list_files[2:]]

        else:
            patch_list = read_binary(standard_list_files[0], format="i",endian=endian) - 1
            event_list = read_binary(standard_list_files[1], format="i",endian=endian) - 1
            patch_slip, patch_time_list = [read_binary(fname, format="d",endian=endian) for fname in standard_list_files[2:]]

        unique_events = np.unique(event_list)
        if len(unique_events) == num_events:
                rcat.event_list = event_list
                rcat.patch_list = patch_list
                rcat.patch_slip, rcat.patch_time_list = patch_slip, patch_time_list
        else:
            print("Event list does not match catalogue length. Trying to fix...")
            if not len(unique_events) > num_events:
                raise ValueError("Event list is too short!")
            last_event = rcat.catalogue_df.index[-1]
            short_events = event_list[event_list <= last_event]
            short_patches = patch_list[:len(short_events)]
            short_slip = patch_slip[:len(short_events)]
            short_time = patch_time_list[:len(short_events)]
            rcat.event_list = short_events
            rcat.patch_list = short_patches
            rcat.patch_slip, rcat.patch_time_list = short_slip, short_time

            print("Fixed!")

        return rcat



    @classmethod
    def from_dataframe_and_arrays(cls, dataframe: pd.DataFrame, event_list: np.ndarray, patch_list: np.ndarray,
                                  patch_slip: np.ndarray, patch_time_list: np.ndarray,reproject: List = None):
        """
        Construct a catalogue from a DataFrame and pre-loaded patch arrays.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Per-event catalogue DataFrame.
        event_list : numpy.ndarray
            Flat 1-D integer array of event IDs, one per patch entry.
        patch_list : numpy.ndarray
            Flat 1-D integer array of patch IDs parallel to ``event_list``.
        patch_slip : numpy.ndarray
            Flat 1-D float array of slip magnitudes (m).
        patch_time_list : numpy.ndarray
            Flat 1-D float array of rupture times (s).
        reproject : list or tuple of int, optional
            ``[in_epsg, out_epsg]`` pair for coordinate reprojection.

        Returns
        -------
        RsqSimCatalogue
            Fully populated catalogue.
        """
        assert all([arr.ndim == 1 for arr in [event_list, patch_list, patch_slip, patch_time_list]])
        list_len = event_list.size
        assert all([arr.size == list_len for arr in [patch_list, patch_slip, patch_time_list]])
        assert len(np.unique(event_list)) == len(dataframe), "Number of events in dataframe and lists do not match"
        rcat = cls.from_dataframe(dataframe,reproject=reproject)
        rcat.event_list, rcat.patch_list, rcat.patch_slip, rcat.patch_time_list = [event_list, patch_list,
                                                                                   patch_slip, patch_time_list]
        return rcat

    @classmethod
    def from_csv_and_arrays(cls, prefix: str, read_index: bool = True, reproject: List = None):
        """
        Construct a catalogue from a CSV and companion NumPy array files.

        Parameters
        ----------
        prefix : str
            File path prefix used by :func:`~rsqsim_api.io.read_utils.read_csv_and_array`.
        read_index : bool, optional
            Whether to read the DataFrame index from the CSV.
            Defaults to ``True``.
        reproject : list or tuple of int, optional
            ``[in_epsg, out_epsg]`` pair for coordinate reprojection.

        Returns
        -------
        RsqSimCatalogue
            Fully populated catalogue.
        """
        df, event_ls, patch_ls, slip_ls, time_ls = read_csv_and_array(prefix, read_index=read_index)
        return cls.from_dataframe_and_arrays(df, event_ls, patch_ls, slip_ls, time_ls, reproject=reproject)

    def write_csv_and_arrays(self, prefix: str, directory: str = None, write_index: bool = True):
        """
        Write the catalogue DataFrame and patch arrays to CSV and NumPy files.

        Parameters
        ----------
        prefix : str
            File name prefix for the output files.
        directory : str or None, optional
            Output directory.  Created if it does not exist.  If
            ``None``, files are written to the current directory.
        write_index : bool, optional
            Whether to write the DataFrame index to the CSV.
            Defaults to ``True``.
        """
        assert prefix, "Empty prefix!"
        if directory is not None:
            if not os.path.exists(directory):
                os.mkdir(directory)

        write_catalogue_dataframe_and_arrays(prefix, self, directory=directory, write_index=write_index)

    def first_event(self, fault_model: RsqSimMultiFault):
        """Return the first event in the catalogue as an :class:`~rsqsim_api.catalogue.event.RsqSimEvent`."""
        return self.events_by_number(int(self.catalogue_df.index[0]), fault_model)[0]

    def nth_event(self,fault_model: RsqSimMultiFault,n: int):
        """
        Return the n-th event (1-based) from the catalogue.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch slip distributions.
        n : int
            1-based index of the event to retrieve.

        Returns
        -------
        RsqSimEvent
        """
        assert isinstance(n,int)
        return self.events_by_number(int(self.catalogue_df.index[n-1]), fault_model)[0]

    def first_n_events(self, number_of_events: int, fault_model: RsqSimMultiFault):
        """
        Return the first ``number_of_events`` events from the catalogue.

        Parameters
        ----------
        number_of_events : int
            Number of events to retrieve.
        fault_model : RsqSimMultiFault
            Fault model for resolving patch slip distributions.

        Returns
        -------
        list of RsqSimEvent
        """
        return self.events_by_number(list(self.catalogue_df.index[:number_of_events]), fault_model)

    def all_events(self, fault_model: RsqSimMultiFault):
        """
        Return all events in the catalogue.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch slip distributions.

        Returns
        -------
        list of RsqSimEvent
        """
        return self.events_by_number(list(self.catalogue_df.index), fault_model)


    def event_outlines(self, fault_model: RsqSimMultiFault, event_numbers: Iterable = None):
        """
        Return the Shapely exterior geometries for a set of events.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch slip distributions.
        event_numbers : iterable of int, optional
            Event IDs to retrieve.  If ``None``, all events are used.

        Returns
        -------
        list of shapely.geometry.base.BaseGeometry
            Unary-union patch outlines for each event.
        """
        if event_numbers is not None:
            events = self.events_by_number(event_numbers, fault_model)
        else:
            events = self.all_events(fault_model)
        return [event.exterior for event in events]

    def filter_df(self, min_t0: fint = None, max_t0: fint = None, min_m0: fint = None,
                  max_m0: fint = None, min_mw: fint = None, max_mw: fint = None,
                  min_x: fint = None, max_x: fint = None, min_y: fint = None, max_y: fint = None,
                  min_z: fint = None, max_z: fint = None, min_area: fint = None, max_area: fint = None,
                  min_dt: fint = None, max_dt: fint = None):
        """
        Return a filtered view of the catalogue DataFrame.

        All parameters are optional range constraints.  Values outside
        ``sensible_ranges`` raise a ``ValueError``.

        Parameters
        ----------
        min_t0, max_t0 : float or int, optional
            Origin time bounds (s).
        min_m0, max_m0 : float or int, optional
            Scalar moment bounds (N·m).
        min_mw, max_mw : float or int, optional
            Moment magnitude bounds.
        min_x, max_x, min_y, max_y : float or int, optional
            Horizontal coordinate bounds (NZTM metres).
        min_z, max_z : float or int, optional
            Depth bounds (m, negative downward).
        min_area, max_area : float or int, optional
            Rupture area bounds (m²).
        min_dt, max_dt : float or int, optional
            Duration bounds (s).

        Returns
        -------
        pandas.DataFrame or None
            Filtered DataFrame, or ``None`` if no conditions were
            specified.
        """
        assert isinstance(self.catalogue_df, pd.DataFrame), "Read in data first!"
        conditions_str = ""
        range_checks = [(min_t0, max_t0, "t0"), (min_m0, max_m0, "m0"), (min_mw, max_mw, "mw"),
                        (min_x, max_x, "x"), (min_y, max_y, "y"), (min_z, max_z, "z"),
                        (min_area, max_area, "area"), (min_dt, max_dt, "dt")]

        if all([any([a is not None for a in (min_m0, max_m0)]),
                any([a is not None for a in (min_mw, max_mw)])]):
            print("Probably no need to filter by both M0 and Mw...")

        for range_check in range_checks:
            min_i, max_i, label = range_check
            if any([a is not None for a in (min_i, max_i)]):
                for a in (min_i, max_i):
                    if not any([a is None, isinstance(a, (float, int))]):
                        raise ValueError("Min and max {} should be int or float".format(label))
                sensible_min, sensible_max = sensible_ranges[label]
                if min_i is None:
                    min_i = sensible_min
                if max_i is None:
                    max_i = sensible_max
                sensible_conditions = all([sensible_min <= a <= sensible_max for a in (min_i, max_i)])

                if not sensible_conditions:
                    raise ValueError("{} values should be between {:e} and {:e}".format(label, sensible_min,
                                                                                        sensible_max))

                range_condition_str = "{} >= {:e} & {} < {:e}".format(label, min_i, label, max_i)
                if not conditions_str:
                    conditions_str += range_condition_str
                else:
                    conditions_str += " & "
                    conditions_str += range_condition_str

        if not conditions_str:
            print("No valid conditions... Copying original catalogue")
            return

        trimmed_df = self.catalogue_df[self.catalogue_df.eval(conditions_str)]
        return trimmed_df

    def filter_whole_catalogue(self, min_t0: fint = None, max_t0: fint = None, min_m0: fint = None,
                               max_m0: fint = None, min_mw: fint = None, max_mw: fint = None,
                               min_x: fint = None, max_x: fint = None, min_y: fint = None, max_y: fint = None,
                               min_z: fint = None, max_z: fint = None, min_area: fint = None, max_area: fint = None,
                               min_dt: fint = None, max_dt: fint = None, reset_index: bool = False):
        """
        Return a new catalogue filtered by catalogue-level parameter ranges.

        Parameters are identical to :meth:`filter_df` with an additional
        ``reset_index`` option.

        Parameters
        ----------
        min_t0, max_t0, min_m0, max_m0, min_mw, max_mw, min_x, max_x,
        min_y, max_y, min_z, max_z, min_area, max_area, min_dt, max_dt :
            See :meth:`filter_df`.
        reset_index : bool, optional
            If ``True``, reindex the filtered catalogue starting from 0.

        Returns
        -------
        RsqSimCatalogue
            New catalogue containing only events within the specified
            parameter ranges.
        """
        trimmed_df = self.filter_df(min_t0, max_t0, min_m0, max_m0, min_mw, max_mw, min_x, max_x, min_y, max_y,
                                    min_z, max_z, min_area, max_area, min_dt, max_dt)
        event_indices = np.where(np.in1d(self.event_list, np.array(trimmed_df.index)))[0]
        trimmed_event_ls = self.event_list[event_indices]
        trimmed_patch_ls = self.patch_list[event_indices]
        trimmed_patch_slip = self.patch_slip[event_indices]
        trimmed_patch_time = self.patch_time_list[event_indices]

        if reset_index:
            trimmed_df.reset_index(inplace=True, drop=True)
            unique_indices = np.unique(trimmed_event_ls)
            index_array = np.zeros(trimmed_event_ls.shape, dtype=int)
            for new_i, old_i in enumerate(unique_indices):
                index_array[np.where(trimmed_event_ls == old_i)] = new_i
        else:
            index_array = trimmed_event_ls

        rcat = self.from_dataframe_and_arrays(trimmed_df, event_list=index_array, patch_list=trimmed_patch_ls,
                                              patch_slip=trimmed_patch_slip, patch_time_list=trimmed_patch_time)
        return rcat

    def filter_by_events(self, event_number: int | Iterable[int], reset_index: bool = False):
        """
        Return a new catalogue containing only the specified events.

        Parameters
        ----------
        event_number : int or iterable of int
            One or more event IDs to retain.
        reset_index : bool, optional
            If ``True``, reindex the filtered catalogue from 0.

        Returns
        -------
        RsqSimCatalogue
            Catalogue containing only the selected events.
        """
        if isinstance(event_number, (int, np.int32,np.int64)):
            ev_ls = [event_number]
        else:
            assert isinstance(event_number, abc.Iterable), "Expecting either int or array/list of ints"
            ev_ls = list(event_number)
            assert all([isinstance(a, (int, np.int32,np.int64)) for a in ev_ls])
        trimmed_df = self.catalogue_df.loc[ev_ls]
        event_indices = np.where(np.in1d(self.event_list, np.array(trimmed_df.index)))[0]
        trimmed_event_ls = self.event_list[event_indices]
        trimmed_patch_ls = self.patch_list[event_indices]
        trimmed_patch_slip = self.patch_slip[event_indices]
        trimmed_patch_time = self.patch_time_list[event_indices]

        if reset_index:
            trimmed_df.reset_index(inplace=True, drop=True)
            unique_indices = np.unique(trimmed_event_ls)
            index_array = np.zeros(trimmed_event_ls.shape, dtype=int)
            for new_i, old_i in enumerate(unique_indices):
                index_array[np.where(trimmed_event_ls == old_i)] = new_i
            print(index_array)
        else:
            index_array = trimmed_event_ls

        rcat = self.from_dataframe_and_arrays(trimmed_df, event_list=index_array, patch_list=trimmed_patch_ls,
                                              patch_slip=trimmed_patch_slip, patch_time_list=trimmed_patch_time)
        return rcat

    def drop_few_patches(self, fault_model: RsqSimMultiFault, min_patches: int = 3):
        """
        Return a catalogue with events having fewer than ``min_patches`` dropped.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch distributions.
        min_patches : int, optional
            Minimum number of patches required for an event to be
            retained.  Defaults to 3.

        Returns
        -------
        RsqSimCatalogue
            Filtered catalogue.
        """
        event_list = self.events_by_number(self.catalogue_df.index, fault_model, min_patches=min_patches)
        new_ids = [ev.event_id for ev in event_list if len(ev.patches) >= min_patches]
        print(len(event_list), new_ids)

        return self.filter_by_events(new_ids)

    def filter_by_fault(self, fault_or_faults: RsqSimMultiFault | RsqSimSegment | list | tuple,
                        minimum_patches_per_fault: int = None):
        """
        Return a catalogue filtered to events that rupture specified faults.

        An event is included if at least one of its patches belongs to
        any fault in ``fault_or_faults``.

        Parameters
        ----------
        fault_or_faults : RsqSimSegment, RsqSimMultiFault, list, or tuple
            Fault(s) to filter on.
        minimum_patches_per_fault : int or None, optional
            If given, an event is only included if it ruptures at least
            this many patches on at least one of the target faults.

        Returns
        -------
        RsqSimCatalogue or None
            Filtered catalogue, or ``None`` if no matching events found.
        """
        if isinstance(fault_or_faults,RsqSimSegment):
            fault_ls = [fault_or_faults]
        elif isinstance(fault_or_faults,RsqSimMultiFault):
            fault_ls=fault_or_faults.faults
        else:
            fault_ls = list(fault_or_faults)

        if minimum_patches_per_fault is not None:
            assert isinstance(minimum_patches_per_fault, int)
            assert minimum_patches_per_fault > 0

        # Collect all fault numbers
        all_patches = []
        for fault in fault_ls:
            all_patches += list(fault.patch_dic.keys())
        patch_numbers = np.unique(np.array(all_patches))

        #in1d will return true if any value in patch_numbers matches a value in patch_list
        #i.e. don't have to rupture all faults in the list, only 1
        patch_indices = np.where(np.in1d(self.patch_list, patch_numbers))[0]
        selected_events = self.event_list[patch_indices]
        selected_patches = self.patch_list[patch_indices]
        if selected_events.size > 0:
            if minimum_patches_per_fault is not None:
                events_gt_min = []
                for fault in fault_ls:
                    fault_patches = np.array(list(fault.patch_dic.keys()))
                    fault_patch_indices = np.where(np.in1d(selected_patches, fault_patches))[0]
                    fault_event_list = selected_events[fault_patch_indices]
                    events_counter = Counter(fault_event_list)
                    events_sufficient_patches = np.array([ev for ev, count in events_counter.items()
                                                          if count >= minimum_patches_per_fault])
                    events_gt_min += list(events_sufficient_patches)
                event_numbers = np.unique(np.array(events_gt_min))
                event_indices = np.where(np.in1d(self.event_list, event_numbers))[0]

            else:
                event_numbers = np.unique(selected_events)
                event_indices = np.where(np.in1d(self.event_list, event_numbers))[0]
            trimmed_df = self.catalogue_df.loc[event_numbers]

            filtered_cat = self.from_dataframe(trimmed_df)
            filtered_cat.event_list = self.event_list[event_indices]
            filtered_cat.patch_list = self.patch_list[event_indices]
            filtered_cat.patch_slip = self.patch_slip[event_indices]
            filtered_cat.patch_time_list = self.patch_time_list[event_indices]

            return filtered_cat
        else:
            print("No events found on the following faults:")
            for fault in fault_ls:
               print(fault.name)

            return None

    def filter_not_on_fault(self, fault_or_faults: RsqSimMultiFault | RsqSimSegment | list | tuple,
                            minimum_patches_per_fault: int = None):
        """
        Return a catalogue with events that rupture specified faults removed.

        The complement of :meth:`filter_by_fault`: events are rejected if
        they rupture any patch on the target faults.

        Parameters
        ----------
        fault_or_faults : RsqSimSegment, RsqSimMultiFault, list, or tuple
            Fault(s) whose events should be excluded.
        minimum_patches_per_fault : int or None, optional
            If given, an event is only rejected if it ruptures at least
            this many patches on one of the target faults.

        Returns
        -------
        RsqSimCatalogue or None
            Filtered catalogue, or ``None`` if no events remain, or
            ``self`` if no events were on the target faults.
        """
        if isinstance(fault_or_faults,RsqSimSegment):
            fault_ls = [fault_or_faults]
        elif isinstance(fault_or_faults,RsqSimMultiFault):
            fault_ls=fault_or_faults.faults
        else:
            fault_ls = list(fault_or_faults)

        if minimum_patches_per_fault is not None:
            assert isinstance(minimum_patches_per_fault, int)
            assert minimum_patches_per_fault > 0

        # Collect all patches we don't want to be involved
        all_patches = []
        for fault in fault_ls:
            all_patches += list(fault.patch_dic.keys())
        patch_numbers = np.unique(np.array(all_patches))

        #in1d will return true if any value in patch_numbers matches a value in patch_list
        #i.e. don't have to rupture all faults in the list, only 1
        # choose all events where this is the case, then remove them from the catalogue
        patch_indices = np.where(np.in1d(self.patch_list, patch_numbers))[0]
        selected_events = self.event_list[patch_indices]
        selected_patches = self.patch_list[patch_indices]
        if selected_events.size > 0:
            if minimum_patches_per_fault is not None:
                events_gt_min = []
                for fault in fault_ls:
                    fault_patches = np.array(list(fault.patch_dic.keys()))
                    fault_patch_indices = np.where(np.in1d(selected_patches, fault_patches))[0]
                    fault_event_list = selected_events[fault_patch_indices]
                    events_counter = Counter(fault_event_list)
                    events_sufficient_patches = np.array([ev for ev, count in events_counter.items()
                                                          if count >= minimum_patches_per_fault])
                    events_gt_min += list(events_sufficient_patches)

                event_numbers2reject = np.unique(np.array(events_gt_min))
                event_indices = np.where(np.in1d(self.event_list, event_numbers2reject))[0]

            else:
                event_numbers2reject = np.unique(selected_events)
                event_indices = np.where(np.in1d(self.event_list, event_numbers2reject))[0]
            if len(event_indices) > 0:
                trimmed_df = self.catalogue_df.drop(event_numbers2reject)

                filtered_cat = self.from_dataframe(trimmed_df)
                filtered_cat.event_list = np.delete(self.event_list,event_indices)
                filtered_cat.patch_list = np.delete(self.patch_list,event_indices)
                filtered_cat.patch_slip = np.delete(self.patch_slip,event_indices)
                filtered_cat.patch_time_list = np.delete(self.patch_time_list,event_indices)

                return filtered_cat
            else:
                print("No remaining events")
                return None
        else:
            print("No events found which include:")
            for fault in fault_ls:
               print(fault.name)

            return self


    def find_surface_rupturing_events(self,fault_model: RsqSimMultiFault,min_slip: float =0.1, method: str = 'vertex',
                                      n_patches: int = 1, max_depth: float = -1000., n_faults: int =1, write_flt_dict: bool = False,
                                      faults2ignore: [list,str] = 'hikurangi'):
        """
        Find catalogue event IDs that include surface rupture.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model used to resolve patch slip distributions.
        min_slip : float, optional
            Minimum slip (m) for a patch to count as ruptured.
            Defaults to 0.1 m.
        method : str, optional
            ``"vertex"`` (use the shallowest vertex) or ``"centroid"``
            (use the centroid depth) for the depth criterion.
        n_patches : int, optional
            Minimum number of qualifying surface patches per fault.
            Defaults to 1.
        max_depth : float, optional
            Depth threshold (m, negative).  Defaults to -1000 m.
        n_faults : int, optional
            Minimum number of surface-rupturing faults per event.
            Defaults to 1.
        write_flt_dict : bool, optional
            If ``True``, also return a dict mapping event IDs to lists
            of surface-rupturing fault names.
        faults2ignore : list of str or str, optional
            Fault name(s) to exclude.  Defaults to ``"hikurangi"``.

        Returns
        -------
        list of int
            Event IDs with surface rupture.
        dict, optional
            Only returned when ``write_flt_dict=True``; maps event ID
            to the list of surface-rupturing fault names.
        """

        assert method in ['centroid','vertex'],"Method must be centroid or vertex"
        assert max_depth < 0., "depths should be negative"
        if write_flt_dict:
            flt_dict = {}
        surface_ev_ids = []
        for event in self.all_events(fault_model=fault_model):
            surface_faults = event.find_surface_faults(fault_model, min_slip=min_slip, method=method,
                                                       n_patches=n_patches, max_depth=max_depth, faults2ignore=faults2ignore)
            if len(surface_faults) >= n_faults:
                surface_ev_ids.append(event.event_id)
                if write_flt_dict:
                    flt_dict[event.event_id] = surface_faults

        if write_flt_dict:
            return surface_ev_ids, flt_dict
        else:
            return surface_ev_ids
    def find_multi_fault(self,fault_model: RsqSimMultiFault):
        """
        Identify events that rupture more than one named fault segment.

        Note that segmentation is based on the fault model's segment
        names and may not correspond to other multi-fault definitions.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch slip distributions.

        Returns
        -------
        multifault : list of RsqSimEvent
            Events that involve more than one fault segment.
        multi_cat : RsqSimCatalogue
            Sub-catalogue containing only those events.
        """
        multifault = [ev for ev in self.all_events(fault_model) if ev.num_faults > 1]
        # and filter catalogue to just these events
        multifault_ids = [event.event_id for event in multifault]
        multi_cat = self.filter_by_events(multifault_ids)
        return multifault,multi_cat

    def filter_by_region(self, region: Polygon | gpd.GeoSeries, fault_model: RsqSimMultiFault,
                         event_numbers: Iterable = None):
        """
        Filter events by geographic region.

        Not yet implemented.

        Parameters
        ----------
        region : Polygon or geopandas.GeoSeries
            Region geometry (reserved for future use).
        fault_model : RsqSimMultiFault
            Fault model (reserved for future use).
        event_numbers : iterable of int, optional
            Reserved for future use.
        """
        pass

    def filter_by_patch_numbers(self, patch_numbers):
        """
        Return a catalogue containing only events that ruptured specified patches.

        Parameters
        ----------
        patch_numbers : array-like of int
            Patch IDs to filter on; events with at least one matching
            patch are retained.

        Returns
        -------
        RsqSimCatalogue or None
            Filtered catalogue, or ``None`` if no events match.
        """
        patch_indices = np.where(np.in1d(self.patch_list, patch_numbers))[0]
        event_numbers = self.event_list[patch_indices]
        if event_numbers.size:
            trimmed_df = self.catalogue_df.loc[np.unique(event_numbers)]
            filtered_cat = self.from_dataframe(trimmed_df)
            filtered_cat.event_list = event_numbers
            filtered_cat.patch_list = self.patch_list[patch_indices]
            filtered_cat.patch_slip = self.patch_slip[patch_indices]
            filtered_cat.patch_time_list = self.patch_time_list[patch_indices]
            return filtered_cat
        else:
            print("No events found!")
            return

    def events_by_number(self, event_number: int | Iterable[int], fault_model: RsqSimMultiFault,
                         child_processes: int = 0, min_patches: int = 1) -> list[RsqSimEvent]:
        """
        Retrieve one or more events as fully populated :class:`~rsqsim_api.catalogue.event.RsqSimEvent` objects.

        Parameters
        ----------
        event_number : int or iterable of int
            Event ID(s) to retrieve.
        fault_model : RsqSimMultiFault
            Fault model for resolving patch objects and fault associations.
        child_processes : int, optional
            If > 0, distribute event processing across this many child
            processes using shared memory.  Defaults to 0 (serial).
        min_patches : int, optional
            Minimum patches per fault segment for an event to be
            considered.  Defaults to 1.

        Returns
        -------
        list of RsqSimEvent
            Fully populated events in the order they were requested.
        """
        assert isinstance(fault_model,RsqSimMultiFault), "Fault model required"
        if isinstance(event_number, (int, np.int32, np.int64)):
            ev_ls = [event_number]
        else:
            assert isinstance(event_number, abc.Iterable), "Expecting either int or array/list of ints"
            ev_ls = list(event_number)
            assert all([isinstance(a, (int, np.int32, np.int64)) for a in ev_ls])
        out_events = []

        cat_dict = self.catalogue_df.to_dict(orient='index')

        # Stores the first and last index for each event
        unique_events, unique_event_indices = np.unique(self.event_list, return_index=True)
        unique_dic = {unique_events[i]: (unique_event_indices[i], unique_event_indices[i + 1]) for i in
                      range(len(unique_events) - 1)}
        unique_dic[unique_events[-1]] = (unique_event_indices[-1], len(self.event_list))

        if child_processes == 0:
            for index in ev_ls:
                ev_range = unique_dic[index]
                ev_indices = np.arange(ev_range[0], ev_range[1])
                patch_numbers = self.patch_list[ev_indices]
                patch_slip = self.patch_slip[ev_indices]
                patch_time_list = self.patch_time_list[ev_indices]
                ev_data = cat_dict[index]
                event_i = RsqSimEvent.from_earthquake_list(ev_data['t0'], ev_data['m0'], ev_data['mw'], ev_data['x'],
                                                           ev_data['y'], ev_data['z'], ev_data['area'], ev_data['dt'],
                                                           patch_numbers=patch_numbers,
                                                           patch_slip=patch_slip,
                                                           patch_time=patch_time_list,
                                                           fault_model=fault_model, min_patches=min_patches,
                                                           event_id=index)
                out_events.append(event_i)

        else:
            # Using shared data between processes
            event_list = np.ctypeslib.as_ctypes(self.event_list)
            raw_event_list = sharedctypes.RawArray(event_list._type_, event_list)

            patch_list = np.ctypeslib.as_ctypes(self.patch_list)
            raw_patch_list = sharedctypes.RawArray(event_list._type_, patch_list)

            queue = Queue()  # queue to handle processed events

            # much faster to serialize when dealing with numbers instead of objects
            faults_with_patches = {patch_num: seg.segment_number for (patch_num, seg) in
                                   fault_model.faults_with_patches.items()}

            ev_chunks = np.array_split(np.array(ev_ls),
                                       child_processes)  # break events into equal sized chunks for each child process
            processes = []
            for i in range(child_processes):
                p = Process(target=get_mask, args=(
                    ev_chunks[i], min_patches, faults_with_patches, raw_event_list, raw_patch_list, queue))
                p.start()
                processes.append(p)

            num_events = len(ev_ls)
            while len(out_events) < num_events:
                index, ev_indices, mask = queue.get()
                patch_numbers = self.patch_list[ev_indices]
                patch_slip = self.patch_slip[ev_indices]
                patch_time = self.patch_time_list[ev_indices]
                ev_data = cat_dict[index]
                event_i = RsqSimEvent.from_multiprocessing(ev_data['t0'], ev_data['m0'], ev_data['mw'], ev_data['x'],
                                                           ev_data['y'], ev_data['z'], ev_data['area'], ev_data['dt'],
                                                           patch_numbers, patch_slip, patch_time,
                                                           fault_model, mask, event_id=index)
                out_events.append(event_i)

            for p in processes:
                p.join()

        return out_events

    def assign_accumulated_slip(self):
        """
        Compute and cache total accumulated slip per patch.

        Sums ``patch_slip`` over all events for each unique patch ID
        and stores the result in :attr:`_accumulated_slip`.  Subsequent
        accesses via :attr:`accumulated_slip` return the cached value.
        """
        accumulated_slip = {}
        for patch_i in np.unique(self.patch_list):
            matching = (self.patch_list == patch_i)
            accumulated_slip_i = self.patch_slip[matching].sum()
            accumulated_slip[patch_i] = accumulated_slip_i
        self._accumulated_slip = accumulated_slip

    def assign_event_mean_slip(self, fault_model: RsqSimMultiFault):
        """
        Compute and cache mean slip per event.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch objects.
        """
        event_mean_slip = {}
        for event in self.all_events(fault_model):
            event_mean_slip[event.event_id] = event.mean_slip
        self._event_mean_slip = event_mean_slip

    def assign_event_mean_sdr(self, fault_model: RsqSimMultiFault):
        """
        Compute and cache mean strike, dip, and rake per event.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch objects.
        """
        event_mean_sdr = {}
        for event in self.all_events(fault_model):
            event_mean_sdr[event.event_id] = [round(event.mean_strike),round(event.mean_dip),round(event.mean_rake)]
        self._event_mean_sdr = event_mean_sdr


    def assign_event_length(self, fault_model: RsqSimMultiFault):
        """
        Compute and cache rupture length per event.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch and fault objects.
        """
        event_lengths = {}
        for event in self.all_events(fault_model):
           event.find_length()
           event_lengths[event.event_id] = event.length
        self._event_length = event_lengths


    def plot_accumulated_slip_2d(self, fault_model: RsqSimMultiFault, subduction_cmap: str = "plasma",
                                 crustal_cmap: str = "viridis", show: bool = True,
                                 write: str = None, subplots=None, global_max_sub_slip: int = 0,
                                 global_max_slip: int = 0,
                                 figsize: tuple = (6.4, 4.8), hillshading_intensity: float = 0.0, bounds: tuple = None,
                                 plot_rivers: bool = True, plot_lakes: bool = True,
                                 plot_highways: bool = True, plot_boundaries: bool = False,
                                 create_background: bool = False,
                                 coast_only: bool = True, hillshade_cmap: colors.LinearSegmentedColormap = cm.terrain,
                                 plot_log_scale: bool = False, log_cmap: str = "magma", log_min: float = 1.0,
                                 log_max: float = 100., plot_traces: bool = True, trace_colour: str = "pink",
                                 min_slip_percentile: float = None, min_slip_value: float = None,
                                 plot_zeros: bool = True):
        """
        Plot a 2-D map of accumulated slip across all events in the catalogue.

        Sums slip over the entire catalogue per patch and plots the
        result with separate colourmaps for subduction-interface and
        crustal faults.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model whose patches are coloured by accumulated slip.
        subduction_cmap : str, optional
            Colourmap for subduction-interface patches.  Defaults to
            ``"plasma"``.
        crustal_cmap : str, optional
            Colourmap for crustal patches.  Defaults to ``"viridis"``.
        show : bool, optional
            If ``True`` (default), call ``plt.show()``.
        write : str or None, optional
            File path to save the figure.
        subplots : tuple or str or None, optional
            Existing ``(fig, ax)`` or pickled figure to plot on.
        global_max_sub_slip, global_max_slip : float, optional
            Fixed colourbar maxima; 0 means per-data maximum.
        figsize : tuple of float, optional
            Figure size in inches.
        hillshading_intensity : float, optional
            Hillshading intensity.
        bounds : tuple or None, optional
            Map extent ``(x_min, y_min, x_max, y_max)`` in NZTM.
        plot_rivers, plot_lakes, plot_highways, plot_boundaries : bool, optional
            Toggle background map layers.
        create_background : bool, optional
            Render full background map.
        coast_only : bool, optional
            Render coastline only as background.
        hillshade_cmap : LinearSegmentedColormap, optional
            Hillshade colourmap.
        plot_log_scale : bool, optional
            Use logarithmic colour scaling.
        log_cmap, log_min, log_max : optional
            Parameters for log-scale colouring.
        plot_traces : bool, optional
            Plot fault surface traces.
        trace_colour : str, optional
            Colour for fault traces.
        min_slip_percentile : float or None, optional
            Slip percentile threshold below which patches are zeroed.
        min_slip_value : float or None, optional
            Slip value threshold (m) below which patches are zeroed.
        plot_zeros : bool, optional
            If ``True`` (default), plot zero-slip patches.

        Returns
        -------
        list
            Matplotlib ``PolyCollection`` objects for each fault plotted.
        """
        if bounds is None and fault_model.bounds is not None:
            bounds = fault_model.bounds

        if all([min_slip_percentile is not None, min_slip_value is None]):
            min_slip = np.percentile(self.patch_slip, min_slip_percentile)
        else:
            min_slip = min_slip_value

        if subplots is not None:
            if isinstance(subplots, str):
                # Assume pickled figure
                with open(subplots, "rb") as pfile:
                    loaded_subplots = pickle.load(pfile)
                fig, ax = loaded_subplots
            else:
                # Assume matplotlib objects
                fig, ax = subplots
        elif create_background:
            # TODO: add this directory + file to repo data/other_lines/nz-lake-polygons-topo-1250k.shp
            fig, ax = plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                      bounds=bounds, plot_rivers=plot_rivers, plot_lakes=plot_lakes,
                                      plot_highways=plot_highways, plot_boundaries=plot_boundaries,
                                      hillshade_cmap=hillshade_cmap)
        elif coast_only:
            fig, ax = plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                      bounds=bounds, plot_rivers=False, plot_lakes=False, plot_highways=False,
                                      plot_boundaries=False, hillshade_cmap=hillshade_cmap)
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(figsize)

        # Find maximum slip on subduction interface
        max_slip = 0

        colour_dic = {}
        for f_i, fault in enumerate(fault_model.faults):
            if fault.name in bruce_subduction:
                if plot_zeros:
                    colours = np.zeros(fault.patch_numbers.shape)
                else:
                    colours = np.nan * np.ones(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.accumulated_slip.keys():
                        if min_slip is not None:
                            if self.accumulated_slip[patch_id] >= min_slip:
                                colours[local_id] = self.accumulated_slip[patch_id]
                        else:
                            if self.accumulated_slip[patch_id] > 0.:
                                colours[local_id] = self.accumulated_slip[patch_id]

                colour_dic[f_i] = colours
                if np.nanmax(colours) > max_slip:
                    max_slip = np.nanmax(colours)
        max_slip = global_max_sub_slip if global_max_sub_slip > 0 else max_slip

        plots = []

        # Plot subduction interface
        subduction_list = []
        subduction_plot = None
        for f_i, fault in enumerate(fault_model.faults):
            if fault.name in bruce_subduction:
                subduction_list.append(fault.name)
                if plot_log_scale:
                    subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max))
                else:
                    subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=subduction_cmap, vmin=0., vmax=max_slip)
                plots.append(subduction_plot)

        max_slip = 0
        colour_dic = {}
        for f_i, fault in enumerate(fault_model.faults):
            if fault.name not in bruce_subduction:
                if plot_zeros:
                    colours = np.zeros(fault.patch_numbers.shape)
                else:
                    colours = np.nan * np.ones(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.accumulated_slip.keys():
                        if min_slip is not None:
                            if self.accumulated_slip[patch_id] >= min_slip:
                                colours[local_id] = self.accumulated_slip[patch_id]
                        else:
                            if self.accumulated_slip[patch_id] > 0.:
                                colours[local_id] = self.accumulated_slip[patch_id]
                colour_dic[f_i] = colours
                if np.nanmax(colours) > max_slip:
                    max_slip = np.nanmax(colours)
        max_slip = global_max_slip if global_max_slip > 0 else max_slip

        crustal_plot = None
        for f_i, fault in enumerate(fault_model.faults):
            if fault.name not in bruce_subduction:
                if plot_log_scale:
                    crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                facecolors=colour_dic[f_i],
                                                cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max))
                else:
                    crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                facecolors=colour_dic[f_i],
                                                cmap=crustal_cmap, vmin=0., vmax=max_slip)
                plots.append(crustal_plot)

        if any([subplots is None, isinstance(subplots, str)]):
            if plot_log_scale:
                if subduction_list:
                    sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                    sub_cbar.set_label("Slip (m)")
                elif crustal_plot is not None:
                    crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                    crust_cbar.set_label("Slip (m)")
            else:
                if subduction_list:
                    sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                    sub_cbar.set_label("Subduction slip (m)")
                if crustal_plot is not None:
                    crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                    crust_cbar.set_label("Slip (m)")

        plot_coast(ax=ax)

        if write is not None:
            fig.savefig(write, dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)

        if show and subplots is None:
            plt.show()

        return plots

    def plot_mfd(self, plot_type: str = 'differential' , nSamp: int = 1000,
                 window: float = 80.*csts.seconds_per_year, n_bins: int=50, instrumental_path: str = None, inst_year_min: float = 1940, show: bool = True,
                write: str = None, tmin: float = None, tmax: float = None, depth_min: float = None, depth_max: float =None, mmin: float = 4.5, mmax: float = 9.5,
                 plot_corrected_instrumental: bool = True):
        """
        Plot a magnitude–frequency distribution for the catalogue.

        Produces a Gutenberg-Richter plot (annual rates vs Mw) in either
        differential or cumulative form.  Random time-window samples can
        be overlaid to show variability.

        Parameters
        ----------
        plot_type : str, optional
            ``"differential"`` (default) or ``"cumulative"``.
        nSamp : int, optional
            Number of random time-window samples to draw for the
            variability envelope.  Defaults to 1000.
        window : float, optional
            Duration (s) of each random sample window.  Defaults to
            80 years.
        n_bins : int, optional
            Number of magnitude bins.  Defaults to 50.
        instrumental_path : str or None, optional
            Path to an instrumental seismicity CSV file for comparison.
        inst_year_min : float, optional
            Minimum year for the instrumental catalogue.  Defaults to
            1940.
        show : bool, optional
            If ``True`` (default), call ``plt.show()``.
        write : str or None, optional
            File path to save the figure.
        tmin, tmax : float or None, optional
            Time range (s) to use.  Defaults to the catalogue range.
        depth_min, depth_max : float or None, optional
            Depth range (km) for filtering the instrumental catalogue.
        mmin, mmax : float, optional
            Magnitude range for the histogram/plot.  Defaults to
            4.5–9.5.
        plot_corrected_instrumental : bool, optional
            If ``True`` (default), overlay a magnitude-corrected
            cumulative curve for the instrumental catalogue.
        """

        assert plot_type in ['differential','cumulative'], "plot_type must be one of cumulative or differential"
        if instrumental_path is not None:
            assert os.path.exists(instrumental_path), "Path to instrumental catalogue not found"
            # Read in instrumental seismicity
            seismicity = pd.read_csv(instrumental_path, converters={0: lambda s: str(s)}, infer_datetime_format=True)

        #check for parameters and find them from catalogue if not specified
        if tmin is None:
           tmin=self.catalogue_df['t0'].min(axis=0)
        if tmax is None:
           tmax = self.catalogue_df['t0'].max(axis=0)

        # find magnitudes for whole catalogue
        rsqsim_mags = self.catalogue_df["mw"]
        weightsyrs2 = np.ones(len(rsqsim_mags)) * csts.seconds_per_year / (tmax - tmin)

        plt.figure()
        if plot_type == "differential":
            tints = []
            for i in np.arange(1, nSamp, 1):
                tinit = tmin + (rng.uniform() * (tmax - tmin - window))
                tints.append(tinit)
                tfin = tinit + window
                mags = self.catalogue_df["mw"].loc[
                    (self.catalogue_df['t0'] > tinit) & (self.catalogue_df['t0'] <= tfin)]
                weightsyrs = np.ones(len(mags)) * csts.seconds_per_year / window

                plt.hist(mags, n_bins, weights=weightsyrs, range=(mmin, mmax), histtype='stepfilled', log=True,
                         edgecolor='lightgray', facecolor='lightgray', alpha=0.1)
            # plot last sample with legend
            plt.hist(mags, n_bins, weights=weightsyrs, range=(mmin, mmax), histtype='stepfilled', log=True, edgecolor='lightgray',
                     facecolor='lightgray', alpha=0.1, label=f"{window / csts.seconds_per_year:.0f} yr RSQsim samples")
            # plot full histogram
            plt.hist(rsqsim_mags, n_bins, weights=weightsyrs2, range=(mmin, mmax), histtype='step', log=True, label="RSQsim",
                     edgecolor='grey')

            if instrumental_path is not None:
                #trim to region of interest
                x1 = self.catalogue_df['x'].min()
                y1 = self.catalogue_df['y'].min()
                x2 = self.catalogue_df['x'].max()
                y2 = self.catalogue_df['y'].max()

                maskmag = ((seismicity['Mpref'] >= mmin) & (seismicity['lons'] > x1) & (seismicity['lons'] < x2) & (
                            seismicity['lats'] > y1) & (seismicity['lats'] < y2) & (seismicity['year'] >= inst_year_min) & (
                                       seismicity['depths'] > depth_min) & (seismicity['depths'] <= depth_max))
                latest_year = np.max(seismicity['year'].loc[seismicity['year'] >= inst_year_min])
                instrumental_mags = seismicity["Mpref"].loc[maskmag]
                weightsyrs_IM = np.ones(np.size(instrumental_mags)) / np.ptp(
                    seismicity['year'].loc[seismicity['year'] >= inst_year_min])
                plt.hist(instrumental_mags, n_bins, weights=weightsyrs_IM, range=(mmin, mmax), histtype='step',
                         log=True, label=f"{str(inst_year_min)}-{str(latest_year)}, Mw>{mmin:.1f}", edgecolor='b',
                         linewidth=1.5)
                instrumental_b_value = calculate_b_value(instrumental_mags, time_interval_years=(latest_year - inst_year_min), min_mw=4.5, max_mw=8.5)
                print(f"b-value for instrumental catalogue: {instrumental_b_value[0]:.2f}")

            plt.ylim([0.0001, 1000])

        elif plot_type == "cumulative":
            tints = []
            for i in np.arange(1, nSamp, 1):
                tinit = tmin + (rng.uniform() * (tmax - tmin - window))
                tints.append(tinit)
                tfin = tinit + window
                mags = self.catalogue_df["mw"].loc[
                    (self.catalogue_df['t0'] > tinit) & (self.catalogue_df['t0'] <= tfin)]
                mag_sort = np.sort(mags)[::-1]
                mag_cum = np.ones(shape=np.size(mag_sort))
                mag_cum = np.cumsum(mag_cum)
                weightsyrs = np.ones(len(mags)) * csts.seconds_per_year / (tfin - tinit)
                plt.semilogy(mag_sort, mag_cum * weightsyrs, c='lightgray', alpha=0.6, linewidth=0.5)
            plt.semilogy(mag_sort, mag_cum * weightsyrs, c='lightgray', alpha=0.6, linewidth=0.5,
                           label=f"{window / csts.seconds_per_year:.0f} yr samples")
            new_sort = np.sort(rsqsim_mags)[::-1]
            new_cum = np.ones(shape=np.size(new_sort))
            new_cum = np.cumsum(new_cum)
            plt.semilogy(new_sort, new_cum * weightsyrs2, c='grey', linewidth=1, label="RSQsim")

            if instrumental_path is not None:
                # trim to region of interest
                x1 = self.catalogue_df['x'].min()
                y1 = self.catalogue_df['y'].min()
                x2 = self.catalogue_df['x'].max()
                y2 = self.catalogue_df['y'].max()

                maskmag = ((seismicity['Mpref'] >= mmin) & (seismicity['lons'] > x1) & (seismicity['lons'] < x2) & (
                        seismicity['lats'] > y1) & (seismicity['lats'] < y2) & (seismicity['year'] >= inst_year_min) & (
                                   seismicity['depths'] > depth_min) & (seismicity['depths'] <= depth_max))
                latest_year = np.max(seismicity['year'].loc[seismicity['year'] >= inst_year_min])
                instrumental_mags = seismicity["Mpref"].loc[maskmag]
                weightsyrs_IM = np.ones(np.size(instrumental_mags)) / np.ptp(
                    seismicity['year'].loc[seismicity['year'] >= inst_year_min])
                IMsort = np.sort(instrumental_mags)[::-1]
                IMcum = np.ones(shape=np.size(IMsort))
                IMcum = np.cumsum(IMcum)
                plt.semilogy(IMsort, IMcum * weightsyrs_IM[0], c='b', linewidth=1,
                               label=f'{inst_year_min} - {int(latest_year)}, Mw>{mmin:.1f}')
                if plot_corrected_instrumental:
                    correctionFactor = 2 / 3.0 * 1 / 1.5
                    plt.semilogy(IMsort, IMcum * weightsyrs_IM[0] * correctionFactor, c='b', linestyle=':', linewidth=1,
                                   label=f'{inst_year_min} - {int(latest_year)}, Mw>{mmin:.1f} corr')
                xsortNZFull = IMsort[:]
                ysortNZFull = IMcum * weightsyrs_IM[0]
            plt.ylim([0.0001, 100])

        plt.ylabel('N [$yr^{-1}$]')
        plt.xlabel('M')
        plt.tight_layout()
        plt.legend()

        if os.path.exists(os.path.dirname(write)):
            plt.savefig(write, dpi=450)
        else:
            print("write directory not found")
        if show:
            plt.show()

    def plot_depth_hist(self,fault_model: RsqSimMultiFault, n_bins: int = 10, depth_min: float=None, depth_max: float=None, write: str = None, show: bool = True):
        """
        Plot a histogram of hypocentral depths alongside fault-base depths.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model used to extract fault base depths.
        n_bins : int, optional
            Number of depth bins.  Defaults to 10.
        depth_min : float or None, optional
            Minimum depth (km) for the plot.  Defaults to the shallowest
            hypocentre.
        depth_max : float or None, optional
            Maximum depth (km) for the plot.  Defaults to the deepest
            hypocentre.
        write : str or None, optional
            File path to save the figure.
        show : bool, optional
            If ``True`` (default), display the figure.
        """
        if depth_min is None:
            depth_min=-0.001 * self.catalogue_df['z'].max(axis=0)
        if depth_max is None:
            depth_max = -0.001 * self.catalogue_df['z'].min(axis=0)

        fig, ax = plt.subplots(1, 1)
        depths = self.catalogue_df['z'] * -0.001
        ax.hist(depths, n_bins, range=(depth_min, depth_max), orientation="horizontal", label="Hypocentral depths")
        fault_depths = [fault.max_depth * -0.001 for fault in fault_model.faults]
        ax.set_ylim([depth_max, depth_min])
        plt.ylabel("Depth (km)")
        ax.set_xlabel("# earthquakes")
        ax2 = ax.twiny()
        ax2.hist(fault_depths, orientation="horizontal", histtype='step', edgecolor='b',
                            label='Base of faults')
        ax2.set_xlabel("# faults")
        fig.legend(loc="lower right", bbox_to_anchor=(1, 0), bbox_transform=ax2.transAxes)
        if os.path.exists(os.path.dirname(write)):
            fig.savefig(write,dpi=450)
        if show:
            fig.show()

    def plot_mean_slip_vs_mag(self, fault_model: RsqSimMultiFault, show: bool = True,
                              write: str = None, plot_rel: bool = True):
        """
        Scatter-plot mean patch slip vs moment magnitude for all events.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for computing mean slip per event.
        show : bool, optional
            If ``True`` (default), display the figure.
        write : str or None, optional
            File path to save the figure.
        plot_rel : bool, optional
            If ``True`` (default), overlay the best-fit trend and the
            expected 1/3-slope scaling-law line.
        """
        # check mean slip is assigned
        if self.event_mean_slip is None:
            self.assign_event_mean_slip(fault_model)

        # create dictionary of magnitudes and mean slips
        mag_mean_slip = {}
        for event in self.all_events(fault_model):
            mag = event.mw
            mean_slip = self.event_mean_slip[event.event_id]
            mag_mean_slip[mag] = mean_slip

        # convert to data frame for easy plotting
        slip_dict = pd.DataFrame.from_dict(mag_mean_slip, orient='index', columns=['Mean Slip'])
        slip_dict.reset_index(inplace=True)
        slip_dict.rename(columns={"index": "mag"}, inplace=True)
        slip_dict["log_slip"] = np.log10(slip_dict["Mean Slip"])
        # plot
        ax = slip_dict.plot.scatter(x="mag", y="log_slip")
        if plot_rel:
            trend_func=np.polyfit(slip_dict["mag"],slip_dict["log_slip"],1)
            trend_fit=np.poly1d(trend_func)
            plt.plot(slip_dict["mag"],trend_fit(slip_dict["mag"]),"r--",label="y = {}".format(trend_fit))
            offset=(1./3.)*(slip_dict["mag"][0])-slip_dict["log_slip"][0]
            plt.plot(slip_dict["mag"],(1./3.)*(slip_dict["mag"])-offset
                     ,"b--",label="y = 1/3 x - {:.2f}".format(offset))
            plt.legend()
        plt.xlabel("M$_W$")
        plt.ylabel("log$_1$$_0$ (Mean Slip (m))")
        if write is not None:
            plt.savefig(write, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_area_vs_mag(self, fault_model: RsqSimMultiFault, show: bool = True,
                         write: str = None, plot_rel: bool = True):
        """
        Scatter-plot rupture area vs moment magnitude for all events.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for retrieving event areas.
        show : bool, optional
            If ``True`` (default), display the figure.
        write : str or None, optional
            File path to save the figure.
        plot_rel : bool, optional
            If ``True`` (default), overlay the best-fit log-linear trend.
        """
        # create dictionary of magnitudes and areas
        mag_area = {}
        for event in self.all_events(fault_model):
            mag = event.mw
            mag_area[mag] = event.area

        # convert to data frame for easy plotting
        area_dict = pd.DataFrame.from_dict(mag_area, orient='index', columns=['Area'])
        area_dict.reset_index(inplace=True)
        area_dict.rename(columns={"index": "mag"}, inplace=True)
        area_dict["log_area"] = np.log10(area_dict["Area"])
        # plot
        ax = area_dict.plot.scatter(x="mag", y="log_area")
        if plot_rel:
            log_fit=np.polyfit(area_dict["mag"],area_dict["log_area"],1)
            log_func=np.poly1d(log_fit)
            plt.plot(area_dict["mag"], log_func(area_dict["mag"]), 'r--',
                     label="y = {}".format(log_func))
            plt.legend()
        plt.xlabel("M$_W$")
        plt.ylabel("log$_1$$_0$ (Area (m$^2$))")
        if write is not None:
            plt.savefig(write, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def stress_drops(self, stress_c: float = 2.44):
        """
        Compute stress drops for all events.

        Parameters
        ----------
        stress_c : float, optional
            Shape constant (default 2.44 for a circular crack).

        Returns
        -------
        numpy.ndarray
            Stress drop in Pa for each event.
        """
        return calculate_stress_drop(self.m0, self.area, stress_c=stress_c)

    def scaling_c(self):
        """
        Compute the Gutenberg-Richter scaling parameter c for all events.

        Returns
        -------
        numpy.ndarray
            Scaling parameter c (``Mw - log10(area) + 6``) per event.
        """
        return calculate_scaling_c(self.mw, self.area)

    def scaling_summary_statistics(self, stress_c: float = 2.44):
        """
        Return summary statistics of scaling parameters for the catalogue.

        Parameters
        ----------
        stress_c : float, optional
            Shape constant passed to
            :func:`~rsqsim_api.catalogue.utilities.summary_statistics`.

        Returns
        -------
        pandas.Series
            Summary statistics including max Mw and percentiles of c and
            stress drop.
        """
        return summary_statistics(self.catalogue_df, stress_c=stress_c)

    def all_slip_distributions_to_vtk(self, fault_model: RsqSimMultiFault, output_directory: str,
                                      include_zeros: bool = False, min_slip_value: float = None):
        """
        Write VTK slip distribution files for every event in the catalogue.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for resolving patch objects.
        output_directory : str
            Existing directory to write ``event<id>.vtk`` files.
        include_zeros : bool, optional
            If ``True``, include zero-slip patches.  Defaults to
            ``False``.
        min_slip_value : float or None, optional
            Minimum slip threshold (m) passed to
            :meth:`~rsqsim_api.catalogue.event.RsqSimEvent.slip_dist_to_vtk`.
        """
        assert os.path.exists(output_directory), "Make directory before writing VTK"
        for event in self.all_events(fault_model):
            outfile_path = os.path.join(output_directory, f"event{event.event_id}.vtk")
            event.slip_dist_to_vtk(outfile_path, include_zeros=include_zeros,min_slip_value=min_slip_value)

    def calculate_b_value(self, min_mw: float = 0.0, max_mw: float = 10.0, interval=0.1):
        """
        Estimate the Gutenberg-Richter b-value for the catalogue.

        Parameters
        ----------
        min_mw : float, optional
            Minimum magnitude bin centre to include.  Defaults to 0.0.
        max_mw : float, optional
            Maximum magnitude bin centre to include.  Defaults to 10.0.
        interval : float, optional
            Magnitude bin width.  Defaults to 0.1.

        Returns
        -------
        tuple of float
            ``(b_value, intercept)`` from the least-squares fit to the
            log-linear cumulative MFD.
        """
        time_interval = (self.catalogue_df['t0'].max() - self.catalogue_df['t0'].min())/csts.seconds_per_year
        return calculate_b_value(self.catalogue_df["mw"], time_interval, min_mw=min_mw, max_mw=max_mw, interval=interval)


    def calculate_dominant_magnitudes(self, fault_model: RsqSimMultiFault, unfiltered_catalogue = None,
                                      min_for_median: int = 5):
        """
        Compute the dominant (median cumulant) magnitude for each patch.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch indices.
        unfiltered_catalogue : RsqSimCatalogue or None, optional
            If provided, use this catalogue's magnitude arrays (useful
            when the current catalogue is a filtered subset).
        min_for_median : int, optional
            Minimum number of matching events for the median to be
            computed.  Defaults to 5.

        Returns
        -------
        numpy.ndarray
            Dominant magnitude (Mw) for each patch in ``fault_model``.
        """
        patch_indices = np.array(list(fault_model.patch_dic.keys()), dtype=np.int32)
        event_list = self.event_list
        if unfiltered_catalogue is not None:
            catalogue_mws = np.array(unfiltered_catalogue.catalogue_df["mw"])
            catalogue_m0s = np.array(unfiltered_catalogue.catalogue_df["m0"])
        else:
            catalogue_mws = np.array(self.catalogue_df["mw"])
            catalogue_m0s = np.array(self.catalogue_df["m0"])
        event_mws = catalogue_mws[event_list]
        event_m0s = catalogue_m0s[event_list]

        medians = np.zeros_like(patch_indices, dtype=np.float64)

        return self.dominant_magnitudes(patch_indices, self.patch_list, event_mws, event_m0s, min_for_median=min_for_median)

    @staticmethod
    @njit(parallel=True)
    def dominant_magnitudes(fault_patch_indices: np.ndarray[int | np.int32 | np.int64],
                            catalogue_patch_array: np.ndarray[int | np.int32 | np.int64],
                            event_mws: np.ndarray, event_m0s: np.ndarray,
                            min_for_median: int = 5):
        """
        Numba-parallel computation of dominant magnitude per patch.

        For each patch index, finds all matching entries in the catalogue
        and computes the median cumulant magnitude.

        Parameters
        ----------
        fault_patch_indices : numpy.ndarray of int
            All patch IDs in the fault model.
        catalogue_patch_array : numpy.ndarray of int
            Flat array of patch IDs from the catalogue (parallel to
            ``event_list``).
        event_mws : numpy.ndarray of float
            Mw for each entry in ``catalogue_patch_array``.
        event_m0s : numpy.ndarray of float
            M0 (N·m) for each entry in ``catalogue_patch_array``.
        min_for_median : int, optional
            Minimum matching events required to compute the median.
            Patches below this threshold receive 0.  Defaults to 5.

        Returns
        -------
        numpy.ndarray of float
            Dominant magnitude for each patch (0 if insufficient data).
        """
        medians_array = np.zeros_like(fault_patch_indices, dtype=np.float64)

        for i in prange(len(fault_patch_indices)):
            n_matching = np.count_nonzero(catalogue_patch_array == i)
            if n_matching >= min_for_median:
                matching = np.flatnonzero(catalogue_patch_array == i)
                matching_m0s = event_m0s[matching]
                matching_mws = event_mws[matching]
                med_cumulant = median_cumulant(matching_m0s, matching_mws)
                medians_array[i] = med_cumulant

        return medians_array

    def filter_crustal_events(self, fault_model: RsqSimMultiFault,
                                       subduction_names: tuple = ("hikkerm", "puysegur"),
                                       min_crustal_mw: float = 6.0):
        """
        Return event IDs where crustal moment exceeds a magnitude threshold.

        Uses Numba-parallel processing to check each event's crustal
        moment contribution.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch areas and crustal patch numbers.
        subduction_names : tuple of str, optional
            Fault names classified as subduction interface; all other
            patches are considered crustal.
        min_crustal_mw : float, optional
            Minimum crustal moment magnitude threshold.  Defaults to 6.0.

        Returns
        -------
        numpy.ndarray of int
            Event IDs whose crustal moment meets the threshold.
        """
        min_crustal_m0 = mw_to_m0(min_crustal_mw)
        crustal_patch_numbers = fault_model.get_crustal_patch_numbers(subduction_faults=subduction_names)

        event_indices = np.array(list(self.catalogue_df.index), dtype=np.int32)
        all_patch_areas = fault_model.get_patch_areas()
        patch_area_list = all_patch_areas[self.patch_list]

        return self.filter_crustal_events_parallel(event_indices, crustal_patch_numbers, self.event_list,
                                                   self.patch_list, self.patch_slip, patch_area_list, min_crustal_m0)

    @staticmethod
    @njit(parallel=True)
    def filter_crustal_events_parallel(event_indices: np.ndarray[int | np.int32 | np.int64],
                                       crustal_patch_numbers: np.ndarray[int | np.int32 | np.int64],
                                       event_list: np.ndarray[int | np.int32 | np.int64],
                                       patch_list: np.ndarray[int | np.int32 | np.int64],
                                       patch_slip: np.ndarray[float | np.float32 | np.float64],
                                       patch_area_list: np.ndarray[float | np.float32 | np.float64],
                                       min_crustal_m0: float | np.float32 | np.float64):
        """
        Numba-parallel filter: return events with crustal moment >= threshold.

        Parameters
        ----------
        event_indices : numpy.ndarray of int
            All event IDs to check.
        crustal_patch_numbers : numpy.ndarray of int
            Patch IDs classified as crustal (non-subduction).
        event_list : numpy.ndarray of int
            Flat catalogue event-ID array.
        patch_list : numpy.ndarray of int
            Flat catalogue patch-ID array.
        patch_slip : numpy.ndarray of float
            Slip (m) parallel to ``patch_list``.
        patch_area_list : numpy.ndarray of float
            Patch areas (m²) parallel to ``patch_list``.
        min_crustal_m0 : float
            Minimum crustal scalar moment (N·m) threshold.

        Returns
        -------
        numpy.ndarray of int
            Event IDs whose crustal moment meets ``min_crustal_m0``.
        """
        crustal_bool = np.zeros_like(event_indices, dtype=np.int32)
        for i in prange(len(event_indices)):
            event_i = event_indices[i]
            patch_numbers = patch_list[np.where(event_list == event_i)]
            crustal_patches_i = jit_intersect(patch_numbers, crustal_patch_numbers)
            if len(crustal_patches_i) > 0:
                crustal_patch_indices = np.searchsorted(patch_numbers, crustal_patches_i)
                patch_slip_i = patch_slip[patch_numbers]
                patch_areas_i = patch_area_list[patch_numbers]
                crustal_slip_i = patch_slip_i[crustal_patch_indices]
                crustal_areas_i = patch_areas_i[crustal_patch_indices]
                m0 = np.sum(crustal_slip_i * crustal_areas_i) * 3e10
                if m0 >= min_crustal_m0:
                    crustal_bool[i] = 1

        return event_indices[crustal_bool == 1]

    def match_events_to_crustal_nshm_dicts(self, fault_model: RsqSimMultiFault,
                                           event_ids: np.ndarray[int | np.int32 | np.int64],
                                           crustal_patch_dict: dict, crustal_n_dict: dict,
                                           subduction_names: tuple = ("hikkerm", "puysegur"), threshold_proportion: float = 0.5):
        """
        Match events to NSHM subsection indices using a KD-tree patch lookup.

        For each event, determines which NSHM subsection indices are
        ruptured based on the fraction of the subsection's patches that
        slipped.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing crustal patch numbers.
        event_ids : numpy.ndarray of int
            Event IDs to process.
        crustal_patch_dict : dict
            Mapping of NSHM subsection index (int) to a dict with
            ``"triangle_indices"`` giving the RSQSim patch IDs.
        crustal_n_dict : dict
            Mapping of NSHM subsection index (int) to total number of
            RSQSim patches in that subsection.
        subduction_names : tuple of str, optional
            Fault names excluded from the crustal analysis.
        threshold_proportion : float, optional
            Fraction of subsection patches that must be ruptured for the
            subsection to be considered active.  Defaults to 0.5.

        Returns
        -------
        numba.typed.Dict
            Mapping of event ID (int32) to an array of NSHM subsection
            indices (int32) that were ruptured.
        """
        crustal_patch_numbers = fault_model.get_crustal_patch_numbers(subduction_faults=subduction_names)
        crustal_patch_dict_typed = typed.Dict.empty(types.int32, types.int32[:])
        for key in crustal_patch_dict.keys():
            crustal_patch_dict_typed[key] = np.array(crustal_patch_dict[key]["triangle_indices"], dtype=np.int32)
        crustal_n_dict_typed = typed.Dict.empty(types.int32, types.int32)
        for key in crustal_n_dict.keys():
            crustal_n_dict_typed[key] = crustal_n_dict[key]
        return self.match_crustal_nshm_parallel(event_ids, self.event_list, self.patch_list, crustal_patch_dict_typed,
                                                crustal_n_dict_typed, crustal_patch_numbers,
                                                threshold_proportion=threshold_proportion)



    @staticmethod
    @njit
    def match_crustal_nshm_parallel(event_ids: np.ndarray[int | np.int32 | np.int64],
                                    event_list: np.ndarray[int | np.int32 | np.int64],
                                    patch_list: np.ndarray[int | np.int32 | np.int64],
                                    crustal_patch_dict: typed.Dict,
                                    crustal_n_dict: typed.Dict,
                                    crustal_patch_numbers: np.ndarray[int | np.int32 | np.int64],
                                    threshold_proportion: float = 0.5):
        """
        Numba JIT-compiled: map events to NSHM subsection indices.

        Parameters
        ----------
        event_ids : numpy.ndarray of int32
            Events to process.
        event_list : numpy.ndarray of int32
            Flat catalogue event-ID array.
        patch_list : numpy.ndarray of int32
            Flat catalogue patch-ID array.
        crustal_patch_dict : numba.typed.Dict
            Mapping of subsection index to array of RSQSim patch IDs.
        crustal_n_dict : numba.typed.Dict
            Mapping of subsection index to total patch count.
        crustal_patch_numbers : numpy.ndarray of int32
            All crustal patch IDs.
        threshold_proportion : float, optional
            Fraction threshold for subsection inclusion.  Defaults to
            0.5.

        Returns
        -------
        numba.typed.Dict
            Mapping of event ID (int32) to array of active subsection
            indices (int32).
        """
        nshm_patch_dict = typed.Dict.empty(types.int32, types.int32[:])
        nshm_keys = np.array(list(crustal_patch_dict.keys()), dtype=np.int32)
        for event_i in range(len(event_ids)):
            event_id = event_ids[event_i]
            patch_numbers = patch_list[np.where(event_list == event_id)]
            crustal_patches_i = jit_intersect(patch_numbers, crustal_patch_numbers)
            subsection_bool = np.zeros_like(nshm_keys, dtype=np.int32)
            for subsection_i in range(len(nshm_keys)):
                subsection_number = nshm_keys[subsection_i]
                subsection_patches = crustal_patch_dict[subsection_number]
                subsection_patches_i = jit_intersect(crustal_patches_i, subsection_patches)
                if len(subsection_patches_i) > 0:
                    nshm_n_i = crustal_n_dict[subsection_number]
                    if len(subsection_patches_i) / nshm_n_i >= threshold_proportion:
                        subsection_bool[subsection_i] = 1
            nshm_patch_dict[event_id] = nshm_keys[subsection_bool == 1]
            print(event_id)
        return nshm_patch_dict

                                    










def read_bruce(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/shaw2021/rundir4627",
               fault_file: str = "bruce_faults.in", names_file: str = "bruce_names.in",
               catalogue_file: str = "eqs..out"):
    """
    Read a Bruce Shaw RSQSim run directory and return the fault model and catalogue.

    Parameters
    ----------
    run_dir : str
        Path to the RSQSim run directory.
    fault_file : str, optional
        Fault geometry file name.  Defaults to ``"bruce_faults.in"``.
    names_file : str, optional
        Fault names file name.  Defaults to ``"bruce_names.in"``.
    catalogue_file : str, optional
        Catalogue file name.  Defaults to ``"eqs..out"``.

    Returns
    -------
    bruce_faults : RsqSimMultiFault
        Fault model with UTM coordinates transformed to NZTM.
    catalogue : RsqSimCatalogue
        Fully populated catalogue.
    """
    fault_full = os.path.join(run_dir, fault_file)
    names_full = os.path.join(run_dir, names_file)

    assert os.path.exists(fault_full)

    bruce_faults = RsqSimMultiFault.read_fault_file_bruce(fault_full,
                                                          names_full,
                                                          transform_from_utm=True)

    catalogue_full = os.path.join(run_dir, catalogue_file)
    assert os.path.exists(catalogue_full)

    catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(catalogue_full,
                                                              run_dir, "rundir4627")

    return bruce_faults, catalogue


def read_bruce_if_necessary(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/shaw2021/rundir4627",
                            fault_file: str = "bruce_faults.in", names_file: str = "bruce_names.in",
                            catalogue_file: str = "eqs..out", default_faults: str = "bruce_faults",
                            default_cat: str = "catalogue"):
    """
    Read a Bruce Shaw run directory only if the variables are not already in global scope.

    Parameters
    ----------
    run_dir : str
        Path to the RSQSim run directory.
    fault_file, names_file, catalogue_file : str, optional
        File names for faults, fault names, and the catalogue.
    default_faults : str, optional
        Global variable name expected to hold the fault model.
        Defaults to ``"bruce_faults"``.
    default_cat : str, optional
        Global variable name expected to hold the catalogue.
        Defaults to ``"catalogue"``.

    Returns
    -------
    tuple of (RsqSimMultiFault, RsqSimCatalogue) or None
        Returns the fault model and catalogue if they were not already
        defined; otherwise returns ``None``.
    """
    print(globals())
    if not all([a in globals() for a in (default_faults, default_cat)]):
        bruce_faults, catalogue = read_bruce(run_dir=run_dir, fault_file=fault_file, names_file=names_file,
                                             catalogue_file=catalogue_file)
        return bruce_faults, catalogue


def combine_boundaries(bounds1: list, bounds2: list):
    """
    Return the bounding box that encompasses both input bounding boxes.

    Takes the element-wise minimum for the lower two coordinates and the
    element-wise maximum for the upper two, producing a combined extent.

    Parameters
    ----------
    bounds1 : list
        First bounding box as ``[min_x, min_y, max_x, max_y]``.
    bounds2 : list
        Second bounding box in the same format.

    Returns
    -------
    list
        Combined bounding box ``[min_x, min_y, max_x, max_y]``.

    Raises
    ------
    AssertionError
        If ``bounds1`` and ``bounds2`` have different lengths.
    """
    assert len(bounds1) == len(bounds2)
    min_bounds = [min([a, b]) for a, b in zip(bounds1, bounds2)]
    max_bounds = [max([a, b]) for a, b in zip(bounds1, bounds2)]
    return min_bounds[:2] + max_bounds[2:]

def calculate_b_value(magnitudes, time_interval_years, min_mw: float = 0.0, max_mw: float = 10.0, interval = 0.1):
    """
    Estimate the Gutenberg-Richter b-value from a magnitude catalogue.

    Builds a cumulative MFD histogram, normalises by the observation
    interval, and fits a straight line to log₁₀(rate) vs Mw to obtain
    the b-value (negative slope) and the a-value (intercept).

    Parameters
    ----------
    magnitudes : array-like
        Array of moment magnitudes.
    time_interval_years : float
        Observation duration in years used to convert event counts to
        annual rates.
    min_mw : float, optional
        Lower magnitude cutoff for the regression.  Defaults to 0.0.
    max_mw : float, optional
        Upper magnitude cutoff for the regression.  Defaults to 10.0.
    interval : float, optional
        Magnitude bin width.  Defaults to 0.1.

    Returns
    -------
    b_value : float
        Estimated b-value (positive, i.e. the negative of the fitted
        gradient).
    intercept : float
        Log₁₀ a-value (y-intercept of the Gutenberg-Richter fit).
    """
    magnitudes = np.array(magnitudes)
    mfd_bins = np.arange(0., 10. + interval, interval)
    mfd_hist = np.histogram(magnitudes, bins=mfd_bins)
    bin_centres = mfd_bins[:-1] + interval/2
    trimmed_bins = bin_centres[(bin_centres >= min_mw) & (bin_centres <= max_mw + interval)]
    cumulative_mfd = np.array([sum(mfd_hist[0][i:]) for i in range(len(mfd_hist[0]))], dtype=float)
    trimmed_mfd = cumulative_mfd[(bin_centres >= min_mw) & (bin_centres <= max_mw + interval)]
    trimmed_mfd_no_zeros = trimmed_mfd[trimmed_mfd > 0.]
    trimmed_mfd_no_zeros /= time_interval_years
    trimmed_bins_no_zeros = trimmed_bins[trimmed_mfd > 0.]
    grad, intercept = np.polyfit(trimmed_bins_no_zeros, np.log10(trimmed_mfd_no_zeros), 1)
    return -1 * grad, intercept