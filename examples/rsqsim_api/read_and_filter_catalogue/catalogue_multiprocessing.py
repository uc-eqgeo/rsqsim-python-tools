from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
import os
import numpy as np
import time

run_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    catalogue = RsqSimCatalogue.from_csv_and_arrays(
        os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
    bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                        os.path.join(run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
                                                        transform_from_utm=True)

    event_list = np.unique(catalogue.event_list)

    t0 = time.time()
    catalogue.events_by_number(event_list.tolist(), bruce_faults, child_processes = 2)
    t1 = time.time()
    print("Multiprocessing:", t1 - t0)

    t0 = time.time()
    catalogue.events_by_number(event_list.tolist(), bruce_faults)
    t1 = time.time()
    print("Serial:", t1 - t0)
