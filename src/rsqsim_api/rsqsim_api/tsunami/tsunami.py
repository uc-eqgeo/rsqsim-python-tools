"""
Classes and functions for loading, manipulating, and plotting sea-surface displacement (SSD) fields.

Provides :class:`SeaSurfaceDisplacements` for a single event's SSD
grid, :class:`MultiEventSeaSurface` for a collection of events, and
:func:`events_from_ssd_netcdf` for extracting SSD arrays from a
pre-computed netCDF file.
"""
import numpy as np
from collections.abc import Iterable
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from rsqsim_api.io.array_operations import write_gmt_grd, write_tiff
import os
from rsqsim_api.visualisation.utilities import plot_coast


class SeaSurfaceDisplacements:
    """
    Sea-surface displacement grid for a single earthquake event.

    Attributes
    ----------
    event_number : int
        RSQSim event identifier.
    x_range : numpy.ndarray
        1-D easting coordinates (NZTM, m) of the grid.
    y_range : numpy.ndarray
        1-D northing coordinates (NZTM, m) of the grid.
    disps : numpy.ndarray of shape (len(y_range), len(x_range))
        Sea-surface displacement values in metres.
    """

    def __init__(self, event_number: int, x_range: np.ndarray, y_range: np.ndarray, disps: np.ndarray):
        """
        Parameters
        ----------
        event_number : int or numpy.integer
            RSQSim event identifier.
        x_range : array-like
            1-D easting coordinates.
        y_range : array-like
            1-D northing coordinates.
        disps : array-like of shape (ny, nx)
            Displacement values (m).
        """
        assert isinstance(event_number, np.integer)
        self.event_number = event_number
        self.x_range = np.array(x_range, dtype=np.float32)
        self.y_range = np.array(y_range, dtype=np.float32)
        self.disps = np.array(disps, dtype=np.float32)
        self._data_bounds = None
        print(event_number, np.max(disps))

    @classmethod
    def from_netcdf_file(cls, event_id: int, nc_file: str):
        """
        Load a single event's SSD from a pre-computed netCDF file.

        Parameters
        ----------
        event_id : int
            RSQSim event identifier to extract.
        nc_file : str
            Path to the netCDF file produced by
            :func:`~rsqsim_api.tsunami.gf_netcdf.sea_surface_displacements_multi`.

        Returns
        -------
        SeaSurfaceDisplacements
        """
        assert os.path.exists(nc_file)
        with Dataset(nc_file) as dset:
            x_range, y_range, disp_ls = events_from_ssd_netcdf(event_id, dset, get_xy=True)
        return cls(event_id, x_range, y_range, disp_ls[0])

    @property
    def data_bounds(self):
        """
        Bounding box of grid cells where the displacement exceeds 5 mm.

        Returns
        -------
        list of float
            ``[x_min, y_min, x_max, y_max]`` in NZTM metres.
        """
        if self._data_bounds is None:
            self.get_data_bounds()
        return self._data_bounds

    def get_data_bounds(self):
        """Compute and cache :attr:`data_bounds`."""
        y_where, x_where = np.where(np.abs(self.disps) >= 0.005)
        x_min, x_max = [self.x_range[x_where].min(), self.x_range[x_where].max()]
        y_min, y_max = [self.y_range[y_where].min(), self.y_range[y_where].max()]
        self._data_bounds = [x_min, y_min, x_max, y_max]

    def to_grid(self, grid_name: str):
        """
        Write the displacement grid to a GMT-format netCDF grid file.

        Parameters
        ----------
        grid_name : str
            Output file path (typically ``*.grd``).
        """
        write_gmt_grd(self.x_range, self.y_range, self.disps, grid_name)

    def to_tiff(self, tiff_name: str, epsg: int = 2193):
        """
        Write the displacement grid to a GeoTIFF file.

        Parameters
        ----------
        tiff_name : str
            Output file path.
        epsg : int, optional
            EPSG code for the output CRS.  Defaults to 2193 (NZTM).
        """
        write_tiff(tiff_name, self.x_range, self.y_range, self.disps, epsg=epsg)

    def plot_ssd(self, cmap="RdBu_r", show: bool = True, write: str = None, show_coast: bool = True,
                 subplots: tuple = None, show_cbar: bool = True, global_max_ssd: int = 10, bounds: Iterable = None,
                 hide_axes_labels: bool = False):
        """
        Plot the sea-surface displacement grid on a map.

        Parameters
        ----------
        cmap : str, optional
            Colormap name.  Defaults to ``"RdBu_r"``.
        show : bool, optional
            If ``True`` (default), call ``plt.show()``.
        write : str or None, optional
            File path to save the figure.
        show_coast : bool, optional
            If ``True`` (default), overlay the New Zealand coastline.
        subplots : tuple or None, optional
            ``(fig, ax)`` to draw onto.  A new figure is created if
            ``None``.
        show_cbar : bool, optional
            If ``True`` (default), add a colourbar.
        global_max_ssd : int, optional
            Maximum displacement (m) for the colour scale.
            Defaults to 10.
        bounds : iterable or None, optional
            ``[x_min, y_min, x_max, y_max]`` clip bounds.  Defaults
            to :attr:`data_bounds`.
        hide_axes_labels : bool, optional
            If ``True``, hide axis tick labels and ticks.
            Defaults to ``False``.

        Returns
        -------
        list
            List of matplotlib artist objects (currently empty).
        """
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
    """
    Collection of :class:`SeaSurfaceDisplacements` objects for multiple events.

    Attributes
    ----------
    event_ls : list of SeaSurfaceDisplacements
        List of individual event displacement grids.
    event_dic : dict
        Mapping of event number (int) to :class:`SeaSurfaceDisplacements`.
    """

    def __init__(self, events: Iterable[SeaSurfaceDisplacements]):
        """
        Parameters
        ----------
        events : iterable of SeaSurfaceDisplacements
            Event displacement grids to collect.
        """
        self.event_ls = list(events)
        assert all([isinstance(event, SeaSurfaceDisplacements) for event in self.event_ls])
        self.event_dic = {event.event_number: event for event in self.event_ls}

    @classmethod
    def from_netcdf_file(cls, event_ids: int | Iterable[int], nc_file: str):
        """
        Load multiple events from a pre-computed netCDF SSD file.

        Parameters
        ----------
        event_ids : int or iterable of int
            RSQSim event identifiers to extract.
        nc_file : str
            Path to the netCDF file.

        Returns
        -------
        MultiEventSeaSurface
        """
        assert os.path.exists(nc_file)
        with Dataset(nc_file) as dset:
            x_range, y_range, disp_ls = events_from_ssd_netcdf(event_ids, dset, get_xy=True)
        event_ls = [SeaSurfaceDisplacements(ev_id, x_range, y_range, disp) for ev_id, disp in zip(event_ids, disp_ls)]
        return cls(event_ls)

    def to_gmt_grids(self, prefix: str):
        """
        Write each event's SSD to a GMT grid file.

        Parameters
        ----------
        prefix : str
            File-name prefix; each file is named
            ``{prefix}{event_id}.grd``.
        """
        for ev_id, event in self.event_dic.items():
            grid_name = prefix + "{:d}.grd".format(ev_id)
            event.to_grid(grid_name)

    def to_tiffs(self, prefix: str, epsg: int = 2193):
        """
        Write each event's SSD to a GeoTIFF file.

        Parameters
        ----------
        prefix : str
            File-name prefix; each file is named
            ``{prefix}{event_id}.tif``.
        epsg : int, optional
            EPSG code for the output CRS.  Defaults to 2193.
        """
        for ev_id, event in self.event_dic.items():
            grid_name = prefix + "{:d}.tif".format(ev_id)
            event.to_tiff(grid_name, epsg=epsg)

    def plot_2d(self):
        pass


def events_from_ssd_netcdf(event_ids: int | Iterable[int], nc: Dataset, get_xy: bool = True):
    """
    Extract sea-surface displacement arrays from a netCDF SSD dataset.

    Parameters
    ----------
    event_ids : int or iterable of int
        RSQSim event identifier(s) to extract.
    nc : netCDF4.Dataset
        Open netCDF4 dataset with variables ``"event_id"``,
        ``"ssd"``, ``"x"``, and ``"y"``.
    get_xy : bool, optional
        If ``True`` (default), also return the x and y coordinate
        arrays.

    Returns
    -------
    If ``get_xy`` is ``True``:
        x_range : numpy.ndarray
        y_range : numpy.ndarray
        disp_ls : list of numpy.ndarray
    If ``get_xy`` is ``False``:
        disp_ls : list of numpy.ndarray

    Raises
    ------
    AssertionError
        If any requested event ID is not found in the dataset.
    """
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
