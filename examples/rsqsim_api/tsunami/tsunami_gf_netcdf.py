import numpy as np
from rsqsim_api.tsunami.tsunami_multiprocessing import multiprocess_gf_to_hdf
from rsqsim_api.fault.multifault import RsqSimMultiFault
"""
Calculate and store greens functions for vertical displacements, from every patch on a fault model
"""

if __name__ == "__main__":
    faults = RsqSimMultiFault.read_fault_file_bruce("../../../data/shaw_m7/bruce_faults.in",
                                                    "../../../data/shaw_m7/bruce_fault_names.in",
                                                    transform_from_utm=True)
    # All of NZ
    x_data = np.arange(800000, 2200000, 2000)
    y_data = np.arange(4200000, 6020000, 2000)

    multiprocess_gf_to_hdf(faults, x_data, y_data, out_file_prefix="bruce_2km_")
