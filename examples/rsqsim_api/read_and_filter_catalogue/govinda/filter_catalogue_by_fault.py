from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
import fnmatch

import os

run_dir = os.path.dirname(__file__)

bruce_cat = RsqSimCatalogue.from_catalogue_file_and_lists("../../../../data/bruce/rundir4627/eqs..out",
                                                          "../../../../data/bruce/rundir4627", "rundir4627")
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../../data/bruce_m7/bruce_names.in"),
                                                      transform_from_utm=True)

# Find
alpine_fault_segments = [fault for fault in bruce_faults.faults if fnmatch.fnmatch(fault.name, "alpine*")]
alpine_only = bruce_cat.filter_by_fault(alpine_fault_segments)

# Save catalogue using pandas to_csv
alpine_only.catalogue_df.to_csv("alpine_cat.csv")