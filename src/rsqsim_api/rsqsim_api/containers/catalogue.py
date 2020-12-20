from typing import Union, Iterable
from collections import abc, Counter, defaultdict
import os

from matplotlib import pyplot as plt
from multiprocessing import Queue, Process
from multiprocessing.sharedctypes import RawArray
from functools import partial
import operator
import pandas as pd
import numpy as np

from rsqsim_api.containers.fault import RsqSimMultiFault, RsqSimSegment
from rsqsim_api.io.read_utils import read_earthquake_catalogue, read_binary, catalogue_columns, read_csv_and_array
from rsqsim_api.io.write_utils import write_catalogue_dataframe_and_arrays
from rsqsim_api.visualisation.utilities import plot_coast
from rsqsim_api.containers.bruce_shaw_utilities import bruce_subduction

fint = Union[int, float]
sensible_ranges = {"t0": (0, 1.e15), "m0": (1.e13, 1.e24), "mw": (2.5, 10.0),
                   "x": (0, 1.e8), "y": (0, 1.e8), "z": (-1.e6, 0),
                   "area": (0, 1.e12), "dt": (0, 1200)}

list_file_suffixes = (".pList", ".eList", ".dList", ".tList")
extra_file_suffixes = (".dmuList", ".dsigmaList", ".dtauList", ".taupList")

seconds_per_year = 31557600.0


def get_mask(ev_ls, min_patches, faults_with_patches, event_list, patch_list, queue):
    patches = np.asarray(patch_list)
    events = np.asarray(event_list)

    for index in ev_ls:
        ev_indices = np.argwhere(events == index).flatten()
        patch_numbers = patches[ev_indices]
        patches_on_fault = defaultdict(list)
        for i in patch_numbers:
            patches_on_fault[faults_with_patches[i]].append(i)

        mask = np.full(len(patch_numbers), True)
        for fault in patches_on_fault.keys():
            if len(patches_on_fault[fault]) < min_patches:
                patch_on_fault_indices = np.array([np.argwhere(patch_numbers == i)[0][0] for i in patches_on_fault[fault]])
                mask[patch_on_fault_indices] = False

        queue.put( (index, ev_indices, mask)  )

class RsqSimCatalogue:
    def __init__(self):
        # Essential attributes
        self._catalogue_df = None
        self._event_list = None
        self._patch_list = None
        self._patch_time_list = None
        self._patch_slip = None
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

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame):
        rsqsim_cat = cls()
        rsqsim_cat.catalogue_df = dataframe
        return rsqsim_cat

    @classmethod
    def from_catalogue_file(cls, filename: str):
        assert os.path.exists(filename)
        catalogue_df = read_earthquake_catalogue(filename)
        rsqsim_cat = cls.from_dataframe(catalogue_df)
        return rsqsim_cat

    @classmethod
    def from_catalogue_file_and_lists(cls, catalogue_file: str, list_file_directory: str,
                                      list_file_prefix: str, read_extra_lists: bool = False):
        assert os.path.exists(catalogue_file)
        assert os.path.exists(list_file_directory)
        standard_list_files = [os.path.join(list_file_directory, list_file_prefix + suffix)
                               for suffix in list_file_suffixes]
        for fname, suffix in zip(standard_list_files, list_file_suffixes):
            if not os.path.exists(fname):
                raise FileNotFoundError("{} file required to populate event slip distributions".format(suffix))

        # Read in catalogue to dataframe and initiate class instance
        rcat = cls.from_catalogue_file(catalogue_file)
        rcat.patch_list = read_binary(standard_list_files[0], format="i")
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
            print(index_array)
        else:
            index_array = trimmed_event_ls

        rcat = self.from_dataframe_and_arrays(trimmed_df, event_list=index_array, patch_list=trimmed_patch_ls,
                                              patch_slip=trimmed_patch_slip, patch_time_list=trimmed_patch_time)
        return rcat




    def filter_by_fault(self, fault_or_faults: Union[RsqSimMultiFault, RsqSimSegment, list, tuple],
                        minimum_patches_per_fault: int = None):
        if isinstance(fault_or_faults, (RsqSimSegment, RsqSimMultiFault)):
            fault_ls = [fault_or_faults]
        else:
            fault_ls = list(fault_or_faults)

        if minimum_patches_per_fault is not None:
            assert isinstance(minimum_patches_per_fault, int)
            assert minimum_patches_per_fault > 0

        all_patches = []
        for fault in fault_ls:
            all_patches += list(fault.patch_dic.keys())
        patch_numbers = np.unique(np.array(all_patches))

        patch_indices = np.where(np.in1d(self.patch_list, patch_numbers))[0]
        selected_events = self.event_list[patch_indices]
        selected_patches = self.patch_list[patch_indices]
        if selected_events:
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
                event_numbers = selected_events
                event_indices = patch_indices
            trimmed_df = self.catalogue_df.iloc[event_numbers]

            filtered_cat = self.from_dataframe(trimmed_df)
            filtered_cat.event_list = event_numbers
            filtered_cat.patch_list = self.patch_list[event_indices]
            filtered_cat.patch_slip = self.patch_slip[event_indices]
            filtered_cat.patch_time_list = self.patch_time_list[event_indices]

            return filtered_cat
        else:
            print("No events found on the following faults:")
            for fault in fault_or_faults:
                print(fault.name)
            return

    def find_multi_fault(self):
        pass

    def filter_by_bounding_box(self):
        pass

    def events_by_number(self, event_number: Union[int, np.int, Iterable[np.int]], fault_model: RsqSimMultiFault, child_processes: int = 0):
        if isinstance(event_number, (int, np.int)):
            ev_ls = [event_number]
        else:
            assert isinstance(event_number, abc.Iterable), "Expecting either int or array/list of ints"
            ev_ls = list(event_number)
            assert all([isinstance(a, (int, np.int)) for a in ev_ls])

        out_events = []
        min_patches = 50
        df = self.catalogue_df
        if child_processes == 0:
            for index in ev_ls:
                ev_indices = np.argwhere(self.event_list == index).flatten()
                patch_numbers = self.patch_list[ev_indices]
                patch_slip = self.patch_slip[ev_indices]
                patch_time_list = self.patch_time_list[ev_indices]
                event_i = RsqSimEvent.from_earthquake_list(df.t0[index], df.m0[index], df.mw[index], df.x[index],
                                                           df.y[index], df.z[index], df.area[index], df.dt[index],
                                                           patch_numbers=patch_numbers,
                                                           patch_slip=patch_slip,
                                                           patch_time=patch_time_list,
                                                           fault_model=fault_model, min_patches=min_patches,
                                                           event_id=index)
                out_events.append(event_i)
        else:
            # Using shared data between processes
            event_list = np.ctypeslib.as_ctypes(self.event_list)
            raw_event_list = RawArray(event_list._type_, event_list)

            patch_list = np.ctypeslib.as_ctypes(self.patch_list)
            raw_patch_list = RawArray(event_list._type_, patch_list)

            queue = Queue() # queue to handle processed events

            # much faster to serialize when dealing with numbers instead of objects
            faults_with_patches = {patch_num: seg.segment_number for (patch_num, seg) in fault_model.faults_with_patches.items()}

            ev_chunks = np.array_split(np.array(ev_ls), child_processes) # break events into equal sized chunks for each child process
            processes = []
            for i in range(child_processes):
                p = Process(target=get_mask, args=(ev_chunks[i], min_patches, faults_with_patches, raw_event_list, raw_patch_list, queue))
                p.start()
                processes.append(p)

            num_events = len(ev_ls)
            while len(out_events) < num_events:
                index, ev_indices, mask = queue.get()
                patch_numbers = self.patch_list[ev_indices]
                patch_slip = self.patch_slip[ev_indices]
                patch_time = self.patch_time_list[ev_indices]
                event = RsqSimEvent.from_multiprocessing(df.t0[index], df.m0[index], df.mw[index], df.x[index],
                                                        df.y[index], df.z[index], df.area[index], df.dt[index],
                                                        patch_numbers, patch_slip, patch_time,
                                                        fault_model, mask, event_id=index)
                out_events.append(event)

            for p in processes:
                p.join()

        return out_events


class RsqSimEvent:
    def __init__(self):
        # Event ID
        self.event_id = None
        # Origin time
        self.t0 = None
        # Seismic moment and mw
        self.m0 = None
        self.mw = None
        # Hypocentre location
        self.x, self.y, self.z = (None,) * 3
        # Rupture area
        self.area = None
        # Rupture duration
        self.dt = None

        # Parameters for slip distributions
        self.patches = None
        self.patch_slip = None
        self.faults = None
        self.patch_time = None
        self.patch_numbers = None

    @property
    def num_faults(self):
        return len(self.faults)

    @property
    def boundary(self):
        x1 = min([min(fault.vertices[:, 0]) for fault in self.faults])
        y1 = min([min(fault.vertices[:, 1]) for fault in self.faults])
        x2 = max([max(fault.vertices[:, 0]) for fault in self.faults])
        y2 = max([max(fault.vertices[:, 1]) for fault in self.faults])
        return [x1, y1, x2, y2]

    @classmethod
    def from_catalogue_array(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float, event_id: int = None):
        """

        :param t0:
        :param m0:
        :param mw:
        :param x:
        :param y:
        :param z:
        :param area:
        :param dt:
        :param event_id:
        :return:
        """

        event = cls()
        event.t0, event.m0, event.mw, event.x, event.y, event.z = [t0, m0, mw, x, y, z]
        event.area, event.dt = [area, dt]
        event.event_id = None

        return event

    @classmethod
    def from_earthquake_list(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: Union[list, np.ndarray, tuple],
                             patch_slip: Union[list, np.ndarray, tuple],
                             patch_time: Union[list, np.ndarray, tuple],
                             fault_model: RsqSimMultiFault, filter_single_patches: bool = True,
                             min_patches: int = 10, min_slip: Union[float, int] = 1, event_id: int = None):
        event = cls.from_catalogue_array(
            t0, m0, mw, x, y, z, area, dt, event_id=event_id)

        faults_with_patches = fault_model.faults_with_patches
        patches_on_fault = defaultdict(list)
        for i in patch_numbers:
            patches_on_fault[faults_with_patches[i]].append(i)

        mask = np.full(len(patch_numbers), True)
        for fault in patches_on_fault.keys():
            patches_on_this_fault = patches_on_fault[fault]
            if len(patches_on_this_fault) < min_patches:
                patch_on_fault_indices = np.array([np.argwhere(patch_numbers == i)[0][0] for i in patches_on_this_fault])
                mask[patch_on_fault_indices] = False

        event.patch_numbers = patch_numbers[mask]
        event.patch_slip = patch_slip[mask]
        event.patch_time = patch_time[mask]

        if event.patch_numbers.size > 0:
            patchnum_lookup = operator.itemgetter(*(event.patch_numbers))
            event.patches = list(patchnum_lookup(fault_model.patch_dic))
            event.faults = list(set(patchnum_lookup(fault_model.faults_with_patches)))
        else:
            event.patches = []
            event.faults = []

        return event

    @classmethod
    def from_multiprocessing(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: Union[list, np.ndarray, tuple],
                             patch_slip: Union[list, np.ndarray, tuple],
                             patch_time: Union[list, np.ndarray, tuple],
                             fault_model: RsqSimMultiFault, mask: list, event_id: int = None):
        event = cls.from_catalogue_array(
            t0, m0, mw, x, y, z, area, dt, event_id=event_id)
        event.patch_numbers = patch_numbers[mask]
        event.patch_slip = patch_slip[mask]
        event.patch_time = patch_time[mask]

        if event.patch_numbers.size > 0:
            patchnum_lookup = operator.itemgetter(*(event.patch_numbers))
            event.patches = list(patchnum_lookup(fault_model.patch_dic))
            event.faults = list(set(patchnum_lookup(fault_model.faults_with_patches)))
        else:
            event.patches = []
            event.faults = []

        return event


    def plot_slip_2d(self, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", show: bool = True,
                     write: str = None, subplots = None, global_max_sub_slip: int = 0, global_max_slip: int = 0):
        # TODO: Plot coast (and major rivers?)
        assert self.patches is not None, "Need to populate object with patches!"

        if subplots is not None:
            fig, ax = subplots
        else:
            fig, ax = plt.subplots()

        # Find maximum slip for subduction interface

        # Find maximum slip to scale colourbar
        max_slip = 0

        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name in bruce_subduction:
                colours = np.zeros(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.patch_numbers:
                        slip_index = np.argwhere(self.patch_numbers == patch_id)[0]
                        colours[local_id] = self.patch_slip[slip_index]
                colour_dic[f_i] = colours
                if max(colours) > max_slip:
                    max_slip = max(colours)
        max_slip = global_max_sub_slip if global_max_sub_slip > 0 else max_slip

        plots = []

        # Plot subduction interface
        subduction_list = []
        subduction_plot = None
        for f_i, fault in enumerate(self.faults):
            if fault.name in bruce_subduction:
                subduction_list.append(fault.name)
                subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                               facecolors=colour_dic[f_i],
                                               cmap=subduction_cmap, vmin=0, vmax=max_slip)
                plots.append(subduction_plot)

        max_slip = 0
        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name not in bruce_subduction:
                colours = np.zeros(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.patch_numbers:
                        slip_index = np.argwhere(self.patch_numbers == patch_id)[0]
                        colours[local_id] = self.patch_slip[slip_index]
                colour_dic[f_i] = colours
                if max(colours) > max_slip:
                    max_slip = max(colours)
        max_slip = global_max_slip if global_max_slip > 0 else max_slip

        crustal_plot = None
        for f_i, fault in enumerate(self.faults):
            if fault.name not in bruce_subduction:
                crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                            facecolors=colour_dic[f_i],
                                            cmap=crustal_cmap, vmin=0, vmax=max_slip)
                plots.append(crustal_plot)

        if subplots is None:
            if subduction_list:
                sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                sub_cbar.set_label("Subduction slip (m)")
            if crustal_plot is not None:
                crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                crust_cbar.set_label("Slip (m)")

        if subplots is None:
            plot_coast(ax, clip_boundary=self.boundary)
            ax.set_aspect("equal")

        if write is not None:
            fig.savefig(write, dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)

        if show and subplots is None:
            plt.show()

        return plots

    def plot_slip_3d(self):
        pass


def read_bruce(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce/rundir4627",
               fault_file: str = "zfault_Deepen.in", names_file: str = "znames_Deepen.in",
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


def read_bruce_if_necessary(run_dir: str = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce/rundir4627",
                            fault_file: str = "zfault_Deepen.in", names_file: str = "znames_Deepen.in",
                            catalogue_file: str = "eqs..out", default_faults: str = "bruce_faults",
                            default_cat: str = "catalogue"):
    print(globals())
    if not all([a in globals() for a in (default_faults, default_cat)]):
        bruce_faults, catalogue = read_bruce(run_dir=run_dir, fault_file=fault_file, names_file=names_file,
                                             catalogue_file=catalogue_file)
        return bruce_faults, catalogue
