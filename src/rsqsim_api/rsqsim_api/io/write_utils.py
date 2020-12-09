import numpy as np
import pandas as pd
import os


def write_catalogue_dataframe_and_arrays(prefix: str, catalogue, directory: str = None,
                                         write_index: bool = True):
    if directory is not None:
        assert os.path.exists(directory)
        dir_path = directory
    else:
        dir_path = ""

    assert isinstance(prefix, str)
    assert len(prefix) > 0
    if prefix[-1] != "_":
        prefix += "_"
    prefix_path = os.path.join(dir_path, prefix)
    df_file = prefix_path + "catalogue.csv"
    event_file = prefix_path + "events.npy"
    patch_file = prefix_path + "patches.npy"
    slip_file = prefix_path + "slip.npy"
    slip_time_file = prefix_path + "slip_time.npy"

    catalogue.catalogue_df.to_csv(df_file, index=write_index)
    for file, array in zip([event_file, patch_file, slip_file, slip_time_file],
                           [catalogue.event_list, catalogue.patch_list, catalogue.patch_slip,
                            catalogue.patch_time_list]):
        np.save(file, array)