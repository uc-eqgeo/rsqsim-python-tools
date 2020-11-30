import os

from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.containers.catalogue import RsqSimCatalogue

run_dir = "../../../data/bruce_m7"

# Read in faults (not currently commited to repo, but could be)
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "bruce_faults.in"),
                                                      os.path.join(run_dir, "bruce_fault_names.in"),
                                                      transform_from_utm=True)
# Read in catalogue from supplied csv and numpy arrays
catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "bruce_m7_10kyr"))
