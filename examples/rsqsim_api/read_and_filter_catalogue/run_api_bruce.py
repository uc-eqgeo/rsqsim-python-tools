import os

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue

run_dir = "../../../data/bruce/rundir4627"

# Read in faults (not currently commited to repo, but could be)
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "zfault_Deepen.in"),
                                                      os.path.join(run_dir, "znames_Deepen.in"),
                                                      transform_from_utm=True)
# Read in catalogue from supplied csv and numpy arrays
catalogue = RsqSimCatalogue.from_catalogue_file_and_lists()
