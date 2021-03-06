from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault

import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "../../../data/shaw_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/shaw_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../data/shaw_m7/bruce_names.in"),
                                                      transform_from_utm=True)

e0 = catalogue.first_event(bruce_faults)