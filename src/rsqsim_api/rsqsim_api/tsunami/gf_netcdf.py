import numpy as np
import netCDF4
from glob import glob
import multiprocessing as mp
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

class LookupPatch:
    def __init__(self, patch_index: int, dset: netCDF4.Dataset, dset_index: int):
        self.patch_index = patch_index
        self.dset = dset
        self.dset_index = dset_index


def create_lookup_dict(search_string: str):
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
    event, lookup = input_tuple
    event_id = event.event_id
    disp_shape = lookup[0].dset["ssd"][0].data.shape
    disp = np.zeros(disp_shape)
    for slip, patch in zip(event.patch_slip, event.patches):
        patch_gf = lookup[patch.patch_number]
        disp += slip * patch_gf.dset["ssd"][patch_gf.dset_index]
    return event_id, disp


def sea_surface_displacements_multi(event_ls: list, lookup: dict, out_netcdf: str, num_processes: int = None):
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        results = executor.map(sea_surface_displacements, [(event, lookup) for event in event_ls])
    template_dset = lookup[0].dset
    x_data = template_dset["x"][:]
    y_data = template_dset["y"][:]

    event_id_ls = [event.event_id for event in event_ls]

    out_dset = netCDF4.Dataset(out_netcdf, "w")
    for dim, dim_len in zip(["x", "y", "event_id"], [x_data.size, y_data.size, len(event_ls)]):
        out_dset.createDimension(dim, dim_len)
    out_dset.createVariable("x", np.float32, ("x",))
    out_dset.createVariable("y", np.float32, ("y",))
    out_dset.createVariable("event_id", np.int, ("event_id",))
    out_dset.createVariable("ssd", np.float32, ("event_id", "y", "x"))

    result_list = list(results)
    result_id_ls = [result[0] for result in result_list]

    for i, event_id in enumerate(event_id_ls):
        event_index = result_id_ls.index(event_id)
        event_disp = result_list[event_index][1]
        out_dset["event_id"][i] = event_id
        out_dset["ssd"][i] = event_disp

    out_dset.close()

