from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault

import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "../../../data/shaw_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/shaw_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../data/shaw_m7/bruce_names.in"),
                                                      transform_from_utm=True)

m9 = catalogue.events_by_number(588, bruce_faults)[0]
m9.plot_slip_2d(plot_log_scale=True, log_min=1, coast_only=False, create_background=False, show=True)
