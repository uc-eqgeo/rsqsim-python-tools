from typing import Union

import numpy as np
from rsqsim_api.containers.fault import RsqSimMultiFault

class RsqSimCatalogue:
    pass


class RsqSimEvent:
    def __init__(self):
        # Origin time
        self.t0 = None
        # Seismic moment and mw
        self.m0 = None
        self.mw = None
        # Hypocentre location
        self.x, self.y, self.y = (None,) * 3
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









