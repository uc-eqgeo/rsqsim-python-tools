import os
import pickle

import numpy as np
from rsqsim_api.tsunami.tsunami_multiprocessing import multiprocess_gf_to_hdf
from rsqsim_api.containers.fault import RsqSimMultiFault


jkn = pickle.load(open("jordan_kek_needles.pkl", "rb"))

x_data = np.arange(1400000, 1751000, 1000)
y_data = np.arange(5200000, 5601000, 1000)
x_grid, y_grid = np.meshgrid(x_data, y_data)

multiprocess_gf_to_hdf(jkn, x_grid, y_grid, out_file="../test.hdf5")

