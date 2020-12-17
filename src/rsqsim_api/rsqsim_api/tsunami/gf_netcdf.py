import numpy
import netCDF4
from glob import glob


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

