import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Iterable, Union
from netCDF4 import Dataset
from rsqsim_api.io.array_operations import write_gmt_grd, write_tiff
import os


class SeaSurfaceDisplacements:
    def __init__(self, event_number: int, x_range: np.ndarray, y_range: np.ndarray, disps: np.ndarray):
        assert isinstance(event_number, np.int)
        self.event_number = event_number
        self.x_range = np.array(x_range, dtype=np.float32)
        self.y_range = np.array(y_range, dtype=np.float32)
        self.disps = np.array(disps, dtype=np.float32)

    @classmethod
    def from_netcdf_file(cls, event_id: int, nc_file: str):
        assert os.path.exists(nc_file)
        with Dataset(nc_file) as dset:
            x_range, y_range, disp_ls = events_from_ssd_netcdf(event_id, dset, get_xy=True)
        return cls(event_id, x_range, y_range, disp_ls[0])

    def to_grid(self, grid_name: str):
        write_gmt_grd(self.x_range, self.y_range, self.disps, grid_name)

    def to_tiff(self, tiff_name: str, epsg: int = 2193):
        write_tiff(tiff_name, self.x_range, self.y_range, self.disps, epsg=epsg)

class MultiEventSeaSurface:
    def __init__(self, events: Iterable[SeaSurfaceDisplacements]):
        self.event_ls = list(events)
        assert all([isinstance(event, SeaSurfaceDisplacements) for event in self.event_ls])
        self.event_dic = {event.event_number: event for event in self.event_ls}

    @classmethod
    def from_netcdf_file(cls, event_ids: Union[int, Iterable[int]], nc_file: str):
        assert os.path.exists(nc_file)
        with Dataset(nc_file) as dset:
            x_range, y_range, disp_ls = events_from_ssd_netcdf(event_ids, dset, get_xy=True)
        event_ls = [SeaSurfaceDisplacements(ev_id, x_range, y_range, disp) for ev_id, disp in zip(event_ids, disp_ls)]
        return cls(event_ls)

    def to_gmt_grids(self, prefix: str):
        for ev_id, event in self.event_dic.items():
            grid_name = prefix + "{:d}.grd".format(ev_id)
            event.to_grid(grid_name)

    def to_tiffs(self, prefix: str, epsg: int = 2193):
        for ev_id, event in self.event_dic.items():
            grid_name = prefix + "{:d}.tif".format(ev_id)
            event.to_tiff(grid_name, epsg=epsg)
    def plot_2d(self):
        pass



def events_from_ssd_netcdf(event_ids: Union[int, Iterable[int]], nc: Dataset, get_xy: bool = True):
    if isinstance(event_ids, int):
        event_id_ls = np.array([event_ids])
    else:
        event_id_ls = np.array(event_ids)
        assert event_id_ls.size > 0

    disp_ls = []
    dset_event_ids = np.array(nc["event_id"][:])
    for event_id in event_id_ls:
        assert event_id in dset_event_ids, "Event number not found!"
        index = np.where(dset_event_ids == event_id)[0][0]
        disp = nc["ssd"][index]
        disp_ls.append(disp)

    if get_xy:
        x_range = nc["x"][:]
        y_range = nc["y"][:]
        return x_range, y_range, disp_ls

    else:
        return disp_ls



