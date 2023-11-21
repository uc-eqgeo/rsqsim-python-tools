import numpy as np
from typing import Iterable, Union
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from rsqsim_api.io.array_operations import write_gmt_grd, write_tiff
import os
from rsqsim_api.visualisation.utilities import plot_coast


class SeaSurfaceDisplacements:
    def __init__(self, event_number: int, x_range: np.ndarray, y_range: np.ndarray, disps: np.ndarray):
        assert isinstance(event_number, np.integer)
        self.event_number = event_number
        self.x_range = np.array(x_range, dtype=np.float32)
        self.y_range = np.array(y_range, dtype=np.float32)
        self.disps = np.array(disps, dtype=np.float32)
        self._data_bounds = None
        print(event_number, np.max(disps))

    @classmethod
    def from_netcdf_file(cls, event_id: int, nc_file: str):
        assert os.path.exists(nc_file)
        with Dataset(nc_file) as dset:
            x_range, y_range, disp_ls = events_from_ssd_netcdf(event_id, dset, get_xy=True)
        return cls(event_id, x_range, y_range, disp_ls[0])

    @property
    def data_bounds(self):
        if self._data_bounds is None:
            self.get_data_bounds()
        return self._data_bounds

    def get_data_bounds(self):
        y_where, x_where = np.where(np.abs(self.disps) >= 0.005)
        x_min, x_max = [self.x_range[x_where].min(), self.x_range[x_where].max()]
        y_min, y_max = [self.y_range[y_where].min(), self.y_range[y_where].max()]
        self._data_bounds = [x_min, y_min, x_max, y_max]





    def to_grid(self, grid_name: str):
        write_gmt_grd(self.x_range, self.y_range, self.disps, grid_name)

    def to_tiff(self, tiff_name: str, epsg: int = 2193):
        write_tiff(tiff_name, self.x_range, self.y_range, self.disps, epsg=epsg)

    def plot_ssd(self, cmap="RdBu_r", show: bool = True, write: str = None, show_coast: bool = True,
                 subplots: tuple = None, show_cbar: bool = True, global_max_ssd: int = 10, bounds: Iterable = None,
                 hide_axes_labels: bool = False):
        if subplots is not None:
            fig, ax = subplots
        else:
            fig, ax = plt.subplots()

        if bounds is not None:
            bounds_list = list(bounds)
            assert len(bounds_list) == 4
        else:
            bounds_list = self.data_bounds

        cscale = np.nanmax(self.disps)
        if cscale > global_max_ssd:
            cscale = global_max_ssd
        plots = []

        plot = ax.pcolormesh(self.x_range, self.y_range, self.disps, cmap=cmap, vmin=-1 * cscale,
                             vmax=cscale, shading="auto")
        ax.set_aspect("equal")
        ax.set_xlim(bounds_list[0], bounds_list[2])
        ax.set_ylim(bounds_list[1], bounds_list[3])
        if show_coast:
            plot_coast(ax, clip_boundary=bounds_list)
        if show_cbar:
            cbar = fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Uplift (m)")

        if hide_axes_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        if write is not None:
            fig.savefig(write, dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)
        if show:
            plt.show()

        return plots


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



