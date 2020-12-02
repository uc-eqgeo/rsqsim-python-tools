import os
import pickle

import numpy as np
from rsqsim_api.tsunami.tsunami_multiprocessing import multiprocess_gf_to_hdf
from rsqsim_api.containers.fault import RsqSimMultiFault

if __name__ == "__main__":
    faults = RsqSimMultiFault.read_fault_file_bruce("/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce_m7/bruce_faults.in",
                                                    "/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce_m7/bruce_fault_names.in",
                                                    transform_from_utm=True)
    # All of NZ
    x_data = np.arange(800000, 2200000, 2000)
    y_data = np.arange(4200000, 6020000, 2000)
    x_grid, y_grid = np.meshgrid(x_data, y_data)

    multiprocess_gf_to_hdf(faults.faults[0], x_grid, y_grid, out_file="test.hdf5")

