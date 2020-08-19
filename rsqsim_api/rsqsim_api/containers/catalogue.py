from typing import Union
import os

import pandas as pd
import numpy as np

from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.io.read_utils import read_earthquake_catalogue, catalogue_columns

fint = Union[int, float]
sensible_ranges = {"t0": (0, 1.e15), "m0": (1.e13, 1.e24), "mw": (2.5, 10.0),
                   "x": (0, 1.e8), "y": (0, 1.e8), "z": (-1.e6, 0),
                   "area": (0, 1.e12), "dt": (0, 1200)}

class RsqSimCatalogue:
    def __init__(self):
        # Better to have array for searching rather than attributes
        self._catalogue_df = None
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
    def from_catalogue_file_and_lists(cls):
        pass

    def filter_earthquakes(self, min_t0: fint = None, max_t0: fint = None, min_m0: fint = None,
                           max_m0: fint = None, min_mw: fint = None, max_mw: fint = None,
                           min_x: fint = None, max_x: fint = None, min_y: fint = None, max_y: fint = None,
                           min_z: fint = None, max_z: fint = None, min_area: fint = None, max_area: fint = None,
                           min_dt: fint = None, max_dt: fint = None):

        assert isinstance(self.catalogue_df, pd.DataFrame), "Read in data first!"
        conditions_str = ""
        range_checks = [(min_t0, max_t0, "t0"), (min_m0, max_m0, "m0"), (min_mw, max_mw, "m0"),
                        (min_x, max_x, "x"), (min_y, max_y, "y"), (min_z, max_z, "z"),
                        (min_area, max_area, "area"), (min_dt, max_dt, "dt")]

        if all([any([a is not None for a in (min_m0, max_m0)]),
                any([a is not None for a in (min_mw, max_mw)])]):
            print("Probably no need to filter by both M0 and Mw...")

        for range_check in range_checks:
            min_i, max_i, label = range_check
            if any([a is not None for a in (min_i, max_i)]):
                if not all([a is not None for a in (min_i, max_i)]):
                    raise ValueError("Need to provide both max and min {}".format(label))
                if not all([isinstance(a, (int, float)) for a in (min_i, max_i)]):
                    raise ValueError("Min and max {} should be int or float".format(label))
                sensible_min, sensible_max = sensible_ranges[label]
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






class RsqSimEvent:
    def __init__(self):
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

    @classmethod
    def from_catalogue_array(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float):
        """

        :param t0:
        :param m0:
        :param mw:
        :param x:
        :param y:
        :param z:
        :param area:
        :param dt:
        :return:
        """

        event = cls()
        event.t0, event.m0, event.mw, event.x, event.y, event.z = [t0, m0, mw, x, y, z]
        event.area, event.dt = [area, dt]

        return event

    @classmethod
    def from_earthquake_list(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: Union[list, np.ndarray, tuple],
                             patch_slip: Union[list, np.ndarray, tuple],
                             patch_time: Union[list, np.ndarray, tuple],
                             fault_model: RsqSimMultiFault):
        event = cls.from_catalogue_array(t0, m0, mw, x, y, z, area, dt)
        event.patches = np.array(patch_numbers)
        event.patch_slip = np.array(patch_slip)
        event.patch_time = np.array(patch_time)
        event.faults = list(set([fault_model.patch_dic[a] for a in event.patches]))

    def plot_slip_2d(self):
        assert self.patches is not None, "Need to populate object with patches!"
        pass

    def plot_slip_3d(self):
        pass









