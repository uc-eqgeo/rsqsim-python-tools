import os
import pickle

import numpy as np
from rsqsim_api.tsunami.tsunami_multiprocessing import multiprocess_gf_to_hdf
from rsqsim_api.containers.fault import RsqSimMultiFault

if __name__ == "__main__":
    faults = RsqSimMultiFault.read_fault_file_bruce("/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce_m7/bruce_faults.in",
                                                    "/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce_m7/bruce_fault_names.in",
                                                    transform_from_utm=True)

    x_data = np.arange(800000, 1751000, 1000)
    y_data = np.arange(5200000, 5601000, 1000)
    x_grid, y_grid = np.meshgrid(x_data, y_data)

    multiprocess_gf_to_hdf(faults, x_grid, y_grid, out_file="tsunami_gfs.hdf5")

