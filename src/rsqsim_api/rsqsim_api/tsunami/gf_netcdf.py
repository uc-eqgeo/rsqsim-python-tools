"""
Utilities for reading and computing Green's-function sea-surface displacement (SSD) from netCDF files.

Provides :class:`LookupPatch` for indexing individual patch Green's
functions within multi-file netCDF datasets, and functions for
computing and saving per-event SSD grids using superposition of patch
responses.
"""
import numpy as np
import netCDF4
from glob import glob
from concurrent.futures import ThreadPoolExecutor


class LookupPatch:
    """
    Index entry mapping a fault patch to its location within a netCDF Green's-function dataset.

    Attributes
    ----------
    patch_index : int
        Global patch identifier.
    dset : netCDF4.Dataset
        Open netCDF4 dataset containing the Green's-function data for
        this patch.
    dset_index : int
        Local index of this patch within ``dset``.
    """

    def __init__(self, patch_index: int, dset: netCDF4.Dataset, dset_index: int):
        """
        Parameters
        ----------
        patch_index : int
            Global patch identifier.
        dset : netCDF4.Dataset
            Open netCDF4 dataset.
        dset_index : int
            Local index of the patch within ``dset``.
        """
        self.patch_index = patch_index
        self.dset = dset
        self.dset_index = dset_index


def create_lookup_dict(search_string: str):
    """
    Build a patch-lookup dictionary from a set of netCDF Green's-function files.

    Opens all files matching ``search_string`` and constructs a
    mapping from global patch index to :class:`LookupPatch` objects.

    Parameters
    ----------
    search_string : str
        Glob pattern for the netCDF files, e.g.
        ``"/path/to/gf_files/*.nc"``.

    Returns
    -------
    dict
        Mapping of patch index (int) to :class:`LookupPatch`.
    """
    lookup_dict = {}
    nc_list = []
    files = glob(search_string)
    for fname in files:
        nc_list.append(netCDF4.Dataset(fname, "r"))

    for dset in nc_list:
        for local_i, patch_i in enumerate(dset["index"][:]):
            lookup_dict[patch_i] = LookupPatch(patch_i, dset, local_i)

    return lookup_dict


def sea_surface_displacements(input_tuple):
    """
    Compute the sea-surface displacement grid for a single event.

    Superposes the per-patch Green's functions scaled by the
    event's patch slip values.  Returns NaN-filled arrays for events
    whose patches are not all present in the lookup dictionary.

    Parameters
    ----------
    input_tuple : tuple
        ``(event, lookup)`` where ``event`` is an
        :class:`~rsqsim_api.catalogue.event.RsqSimEvent` and
        ``lookup`` is a dict mapping patch index to
        :class:`LookupPatch`.

    Returns
    -------
    event_id : int
        The event identifier.
    disp : numpy.ndarray
        Sea-surface displacement grid (shape matches the Green's
        function grid).  Filled with ``NaN`` if any patch is missing
        from the lookup.
    """
    event, lookup = input_tuple
    event_id = event.event_id
    disp_shape = lookup[0].dset["ssd"][0].data.shape
    disp = np.zeros(disp_shape)
    try:
        for slip, patch in zip(event.patch_slip, event.patches):
            patch_gf = lookup[patch.patch_number]
            disp += slip * patch_gf.dset["ssd"][patch_gf.dset_index]
    except KeyError:
        disp = np.ones(disp_shape) * np.nan

    return event_id, disp


def sea_surface_displacements_multi(event_ls: list, lookup: dict, out_netcdf: str, num_processes: int = None):
    """
    Compute and save sea-surface displacements for multiple events to a netCDF file.

    Uses a :class:`~concurrent.futures.ThreadPoolExecutor` to compute
    displacements in parallel, then writes the results to a new netCDF
    file with dimensions ``(event_id, y, x)``.

    Parameters
    ----------
    event_ls : list
        List of :class:`~rsqsim_api.catalogue.event.RsqSimEvent`
        objects.
    lookup : dict
        Patch-lookup dictionary from :func:`create_lookup_dict`.
    out_netcdf : str
        Output netCDF file path.
    num_processes : int or None, optional
        Maximum number of worker threads.  Defaults to ``None``
        (determined by :class:`~concurrent.futures.ThreadPoolExecutor`
        default).
    """
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(sea_surface_displacements, [(event, lookup) for event in event_ls])
    template_dset = lookup[0].dset
    x_data = template_dset["x"][:]
    y_data = template_dset["y"][:]

    event_id_ls = [event.event_id for event in event_ls]

    out_dset = netCDF4.Dataset(out_netcdf, "w")
    out_dset.set_always_mask(False)
    for dim, dim_len in zip(["x", "y", "event_id"], [x_data.size, y_data.size, len(event_ls)]):
        out_dset.createDimension(dim, dim_len)
    out_dset.createVariable("x", np.float32, ("x",))
    out_dset.createVariable("y", np.float32, ("y",))
    out_dset["x"][:] = x_data
    out_dset["y"][:] = y_data
    out_dset.createVariable("event_id", int, ("event_id",))
    out_dset.createVariable("ssd", np.float32, ("event_id", "y", "x"), zlib=True)

    result_list = list(results)
    result_id_ls = [result[0] for result in result_list]

    num_events = len(event_id_ls)
    for i, event_id in enumerate(event_id_ls):
        print(f"{i}/{num_events}")
        event_index = result_id_ls.index(event_id)
        event_disp = result_list[event_index][1]
        out_dset["event_id"][i] = event_id
        out_dset["ssd"][i] = event_disp

    out_dset.close()
