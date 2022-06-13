from typing import Union, Iterable, List, Dict, Any
from collections import abc, Counter, defaultdict
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

from rsqsim_api.fault.multifault import RsqSimMultiFault, RsqSimSegment
from rsqsim_api.catalogue.event import RsqSimEvent
from rsqsim_api.io.read_utils import read_earthquake_catalogue, read_binary, catalogue_columns, read_csv_and_array
from rsqsim_api.io.write_utils import write_catalogue_dataframe_and_arrays
from rsqsim_api.tsunami.tsunami import SeaSurfaceDisplacements
from rsqsim_api.visualisation.utilities import plot_coast, plot_background, plot_hillshade_niwa, plot_lake_polygons, \
    plot_river_lines, plot_highway_lines, plot_boundary_polygons
from rsqsim_api.io.bruce_shaw_utilities import bruce_subduction
import rsqsim_api.io.rsqsim_constants as csts

fint = Union[int, float]
sensible_ranges = {"t0": (0, 1.e15), "m0": (1.e13, 1.e24), "mw": (2.5, 10.0),
                   "x": (-180., 1.e8), "y": (-90., 1.e8), "z": (-1.e6, 0),
                   "area": (0, 1.e12), "dt": (0, 1200)}

list_file_suffixes = (".pList", ".eList", ".dList", ".tList")
extra_file_suffixes = (".dmuList", ".dsigmaList", ".dtauList", ".taupList")


def get_mask(ev_ls, min_patches, faults_with_patches, event_list, patch_list, queue):
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
    def __init__(self):
        # Essential attributes
        self._catalogue_df = None
        self._event_list = None
        self._patch_list = None
        self._patch_time_list = None
        self._patch_slip = None
        self._accumulated_slip = None
        self._event_mean_slip = None
        self._event_mean_sdr = None
        self._event_length = None
        # Useful attributes
        self.t0, self.m0, self.mw = (None,) * 3
        self.x, self.y, self.z = (None,) * 3
        self.area, self.dt = (None,) * 2

    @property
    def catalogue_df(self):
        return self._catalogue_df

    @catalogue_df.setter
    def catalogue_df(self, dataframe: pd.DataFrame):
        assert dataframe.columns.size == 8, "Should have 8 columns"
        assert all([col.dtype in ("float", "int") for i, col in dataframe.iteritems()])
        dataframe.columns = catalogue_columns
        self._catalogue_df = dataframe

    def check_list(self, data_list: np.ndarray, data_type: str):
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
        return self._event_list

    @event_list.setter
    def event_list(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="i")
        if not len(np.unique(data_list)) == len(self.catalogue_df):
            raise ValueError("Numbers of events in catalogue and supplied list are different!")
        self._event_list = data_list

    @property
    def patch_list(self):
        return self._patch_list

    @patch_list.setter
    def patch_list(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="i")
        self._patch_list = data_list

    @property
    def patch_time_list(self):
        return self._patch_time_list

    @patch_time_list.setter
    def patch_time_list(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="d")
        self._patch_time_list = data_list

    @property
    def patch_slip(self):
        return self._patch_slip

    @patch_slip.setter
    def patch_slip(self, data_list: np.ndarray):
        self.check_list(data_list, data_type="d")
        self._patch_slip = data_list

    @property
    def accumulated_slip(self):
        if self._accumulated_slip is None:
            self.assign_accumulated_slip()
        return self._accumulated_slip

    @property
    def event_mean_slip(self):
        return self._event_mean_slip

    @property
    def event_mean_sdr(self):
        return self._event_mean_sdr

    @property
    def event_length(self):
        return self._event_length

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, reproject: List = None):
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
        assert os.path.exists(filename)
        catalogue_df = read_earthquake_catalogue(filename)
        rsqsim_cat = cls.from_dataframe(catalogue_df, reproject=reproject)
        return rsqsim_cat

    @classmethod
    def from_catalogue_file_and_lists(cls, catalogue_file: str, list_file_directory: str,
                                      list_file_prefix: str, read_extra_lists: bool = False, reproject: List = None):
        assert os.path.exists(catalogue_file)
        assert os.path.exists(list_file_directory)

        standard_list_files = [os.path.join(list_file_directory, list_file_prefix + suffix)
                               for suffix in list_file_suffixes]
        for fname, suffix in zip(standard_list_files, list_file_suffixes):
            if not os.path.exists(fname):
                raise FileNotFoundError("{} file required to populate event slip distributions".format(suffix))

        # Read in catalogue to dataframe and initiate class instance
        rcat = cls.from_catalogue_file(catalogue_file, reproject=reproject)
        rcat.patch_list = read_binary(standard_list_files[0], format="i") - 1
        # indices start from 1, change so that it is zero instead
        rcat.event_list = read_binary(standard_list_files[1], format="i") - 1
        rcat.patch_slip, rcat.patch_time_list = [read_binary(fname, format="d") for fname in standard_list_files[2:]]

        return rcat

    @classmethod
    def from_dataframe_and_arrays(cls, dataframe: pd.DataFrame, event_list: np.ndarray, patch_list: np.ndarray,
                                  patch_slip: np.ndarray, patch_time_list: np.ndarray):
        assert all([arr.ndim == 1 for arr in [event_list, patch_list, patch_slip, patch_time_list]])
        list_len = event_list.size
        assert all([arr.size == list_len for arr in [patch_list, patch_slip, patch_time_list]])
        assert len(np.unique(event_list)) == len(dataframe), "Number of events in dataframe and lists do not match"
        rcat = cls.from_dataframe(dataframe)
        rcat.event_list, rcat.patch_list, rcat.patch_slip, rcat.patch_time_list = [event_list, patch_list,
                                                                                   patch_slip, patch_time_list]
        return rcat

    @classmethod
    def from_csv_and_arrays(cls, prefix: str, read_index: bool = True):
        df, event_ls, patch_ls, slip_ls, time_ls = read_csv_and_array(prefix, read_index=read_index)
        return cls.from_dataframe_and_arrays(df, event_ls, patch_ls, slip_ls, time_ls)

    def write_csv_and_arrays(self, prefix: str, directory: str = None, write_index: bool = True):
        assert prefix, "Empty prefix!"
        write_catalogue_dataframe_and_arrays(prefix, self, directory=directory, write_index=write_index)

    def first_event(self, fault_model: RsqSimMultiFault):
        return self.events_by_number(int(self.catalogue_df.index[0]), fault_model)[0]

    def nth_event(self,fault_model: RsqSimMultiFault,n: int):
        assert isinstance(n,int)
        return self.events_by_number(int(self.catalogue_df.index[n-1]), fault_model)[0]

    def first_n_events(self, number_of_events: int, fault_model: RsqSimMultiFault):
        return self.events_by_number(list(self.catalogue_df.index[:number_of_events]), fault_model)

    def all_events(self, fault_model: RsqSimMultiFault):
        return self.events_by_number(list(self.catalogue_df.index), fault_model)

    def all_events_generator(self, fault_model: RsqSimMultiFault):
        return self.events_by_number(list(self.catalogue_df.index), fault_model, generator=True)

    def event_outlines(self, fault_model: RsqSimMultiFault, event_numbers: Iterable = None):
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
            index_array = np.zeros(trimmed_event_ls.shape, dtype=np.int)
            for new_i, old_i in enumerate(unique_indices):
                index_array[np.where(trimmed_event_ls == old_i)] = new_i
        else:
            index_array = trimmed_event_ls

        rcat = self.from_dataframe_and_arrays(trimmed_df, event_list=index_array, patch_list=trimmed_patch_ls,
                                              patch_slip=trimmed_patch_slip, patch_time_list=trimmed_patch_time)
        return rcat

    def filter_by_events(self, event_number: Union[int, np.int, Iterable[int]], reset_index: bool = False):
        if isinstance(event_number, (int, np.int)):
            ev_ls = [event_number]
        else:
            assert isinstance(event_number, abc.Iterable), "Expecting either int or array/list of ints"
            ev_ls = list(event_number)
            assert all([isinstance(a, (int, np.int)) for a in ev_ls])
        trimmed_df = self.catalogue_df.loc[ev_ls]
        event_indices = np.where(np.in1d(self.event_list, np.array(trimmed_df.index)))[0]
        trimmed_event_ls = self.event_list[event_indices]
        trimmed_patch_ls = self.patch_list[event_indices]
        trimmed_patch_slip = self.patch_slip[event_indices]
        trimmed_patch_time = self.patch_time_list[event_indices]

        if reset_index:
            trimmed_df.reset_index(inplace=True, drop=True)
            unique_indices = np.unique(trimmed_event_ls)
            index_array = np.zeros(trimmed_event_ls.shape, dtype=np.int)
            for new_i, old_i in enumerate(unique_indices):
                index_array[np.where(trimmed_event_ls == old_i)] = new_i
            print(index_array)
        else:
            index_array = trimmed_event_ls

        rcat = self.from_dataframe_and_arrays(trimmed_df, event_list=index_array, patch_list=trimmed_patch_ls,
                                              patch_slip=trimmed_patch_slip, patch_time_list=trimmed_patch_time)
        return rcat

    def drop_few_patches(self, fault_model: RsqSimMultiFault, min_patches: int = 3):
        event_list = self.events_by_number(self.catalogue_df.index, fault_model, min_patches=min_patches)
        new_ids = [ev.event_id for ev in event_list if len(ev.patches) >= min_patches]
        print(len(event_list), new_ids)

        return self.filter_by_events(new_ids)

    def filter_by_fault(self, fault_or_faults: Union[RsqSimMultiFault, RsqSimSegment, list, tuple],
                        minimum_patches_per_fault: int = None):
        if isinstance(fault_or_faults, (RsqSimSegment, RsqSimMultiFault)):
            fault_ls = [fault_or_faults]
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
            for fault in fault_or_faults:
                print(fault.name)
            return

    def find_multi_fault(self,fault_model: RsqSimMultiFault):
        """
        Identify events involving more than 1 fault. Note that this is based on named fault segments so might not
        reflect the area ruptured/ be consistent with other approaches to understanding multifault ruptures.

        Parameters
        ----------
        fault_model : RsqSimMultiFault object

        Returns
        -------
        list of multifault events
        RsqSimCatalogue with only multifault ruptures
        """
        multifault = [ev for ev in self.all_events(fault_model) if ev.num_faults > 1]
        # and filter catalogue to just these events
        multifault_ids = [event.event_id for event in multifault]
        multi_cat = self.filter_by_events(multifault_ids)
        return multifault,multi_cat


    def find_single_fault(self,fault_model: RsqSimMultiFault):
        """
        Identify events involving only 1 fault. Note that this is based on named fault segments so might not
        reflect the area ruptured/ be consistent with other approaches to understanding multifault ruptures.

        Parameters
        ----------
        fault_model : RsqSimMultiFault object

        Returns
        -------
        list of single fault events
        RsqSimCatalogue with only single ault ruptures
        """
        singlefault = [ev for ev in self.all_events(fault_model) if ev.num_faults == 1]
        # and filter catalogue to just these events
        singlefault_ids = [event.event_id for event in singlefault]
        single_cat = self.filter_by_events(singlefault_ids)
        return singlefault,single_cat

    def filter_by_region(self, region: Union[Polygon, gpd.GeoSeries], fault_model: RsqSimMultiFault,
                         event_numbers: Iterable = None):
        pass

    def filter_by_patch_numbers(self, patch_numbers):
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

    def events_by_number(self, event_number: Union[int, np.int, Iterable[np.int]], fault_model: RsqSimMultiFault,
                         child_processes: int = 0, min_patches: int = 1):
        if isinstance(event_number, (int, np.int)):
            ev_ls = [event_number]
        else:
            assert isinstance(event_number, abc.Iterable), "Expecting either int or array/list of ints"
            ev_ls = list(event_number)
            assert all([isinstance(a, (int, np.int)) for a in ev_ls])

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
        Create dict of patch numbers with slip accumulated over events in the catalogue.
        Note that this overwrites any other value which could have been assigned to accumulated slip.
        """
        accumulated_slip = {}
        for patch_i in np.unique(self.patch_list):
            matching = (self.patch_list == patch_i)
            accumulated_slip_i = self.patch_slip[matching].sum()
            accumulated_slip[patch_i] = accumulated_slip_i
        self._accumulated_slip = accumulated_slip

    def assign_event_mean_slip(self, fault_model: RsqSimMultiFault):
        """
        Create dict of event ids with associated mean slip on the patches which slip in them.
        Note that this overwrites any other value which could have been assigned to mean slip.
        """
        event_mean_slip = {}
        for event in self.all_events(fault_model):
            event_mean_slip[event.event_id] = event.mean_slip
        self._event_mean_slip = event_mean_slip

    def assign_event_mean_sdr(self, fault_model: RsqSimMultiFault):
        """
        Create dict of event ids with associated mean strike,dip and rake on the patches which slip in them.
        Note that this overwrites any other value which could have been assigned to mean strike, mean dip or mean rake.
        """
        event_mean_sdr = {}
        for event in self.all_events(fault_model):
            event_mean_sdr[event.event_id] = [round(event.mean_strike),round(event.mean_dip),round(event.mean_rake)]
        self._event_mean_sdr = event_mean_sdr


    def assign_event_length(self, fault_model: RsqSimMultiFault):
        """
        Create dict of event ids with associated maximum horizontal straight line distances between patches which slip in them.
        Note that this overwrites any other value which could have been assigned to event length.
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

    def plot_gr(self, fault_model: RsqSimMultiFault, show: bool = True,
                write: str = None, best_fit: bool = True):
        """
        Plot Gutenburg-Richter distribution for a given catalogue.
        y-axis: log(annual frequency of events with M>Mw)
        x axis: Mw

        Parameters
        ----------
        fault_model : RsqSimMultiFault object
        show : bool to indicate whether to display plot
        write : file (if any) to write plot to
        best_fit : whether to plot best fit linear trend w/b value
        """
        mag_freq: dict[int, float | Any] = {}
        # find min magnitude of catalogue
        mws = np.array([ev.mw for ev in self.all_events(fault_model)])
        min_mag = np.floor(np.min(mws))
        max_mag = np.max(mws)

        for mag in np.arange(min_mag, max_mag, 0.5):

            filt_cat = self.filter_whole_catalogue(min_mw=mag)
            print("Filtering for {}\n".format(mag))
            if filt_cat.event_list.size != 0:

                nevents = len(filt_cat.all_events(fault_model))

                times = np.array([ev.t0 for ev in filt_cat.all_events(fault_model)])
                tot_time = (np.max(times) - np.min(times)) / csts.seconds_per_year
                freq = nevents / tot_time
            else:
                freq = 0
            mag_freq[mag] = freq
        # convert to dataframe for plotting
        mf_dict = pd.DataFrame.from_dict(mag_freq, orient='index', columns=['freq'])
        mf_dict.reset_index(inplace=True)
        mf_dict.rename(columns={"index": "mag"}, inplace=True)
        mf_dict["log_freq"] = np.log10(mf_dict["freq"])

        # plot
        print("plotting")
        ax = mf_dict.plot.scatter(x="mag", y="freq")
        if best_fit:
            gr_fit=np.polyfit(mf_dict['mag'],mf_dict['log_freq'],1)
            gr_trend=np.poly1d(gr_fit)
            plt.plot(mf_dict["mag"],10**gr_trend(mf_dict["mag"]),'r--',label="b="+str(round(abs(gr_trend.c[0]),2)))
            ax.legend()
        ax.set_yscale('log')
        ax.set_xticks(np.arange(min_mag, 10, 1))
        plt.xlabel("Mw")
        plt.ylabel("# events with M>Mw per year")

        if write is not None:
            plt.savefig(write, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_mean_slip_vs_mag(self, fault_model: RsqSimMultiFault, show: bool = True,
                              write: str = None, plot_rel: bool = True):
        """
        Plot the mean slip on each patch against the moment magnitude of the earthquake for each event in catalogue.

        Parameters
        -------
        fault_model : RsqSimMultiFault object
        show : bool to indicate whether to display plot
        write : file (if any) to write plot to
        plot_rel : plot expected analytic relationship best on scaling laws
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
        Plot the area of each event against the seismic moment of the earthquake for each event in catalogue.

        Parameters
        -------

        fault_model : RsqSimMultiFault object
        show : bool to indicate whether to display plot
        write : file (if any) to write plot to
        plot_rel : plot best fit logarithmic trend

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

    def all_slip_distributions_to_vtk(self, fault_model: RsqSimMultiFault, output_directory: str,
                                      include_zeros: bool = False):
        """

        @param fault_model:
        @param output_directory:
        @param include_zeros:
        @return:
        """
        assert os.path.exists(output_directory), "Make directory before writing VTK"
        for event in self.all_events_generator(fault_model):
            outfile_path = os.path.join(output_directory, f"event{event.event_id}.vtk")
            event.slip_dist_to_vtk(outfile_path, include_zeros=include_zeros)






def read_bruce(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/shaw2021/rundir4627",
               fault_file: str = "bruce_faults.in", names_file: str = "bruce_names.in",
               catalogue_file: str = "eqs..out"):
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
    print(globals())
    if not all([a in globals() for a in (default_faults, default_cat)]):
        bruce_faults, catalogue = read_bruce(run_dir=run_dir, fault_file=fault_file, names_file=names_file,
                                             catalogue_file=catalogue_file)
        return bruce_faults, catalogue


def combine_boundaries(bounds1: list, bounds2: list):
    assert len(bounds1) == len(bounds2)
    min_bounds = [min([a, b]) for a, b in zip(bounds1, bounds2)]
    max_bounds = [max([a, b]) for a, b in zip(bounds1, bounds2)]
    return min_bounds[:2] + max_bounds[2:]
