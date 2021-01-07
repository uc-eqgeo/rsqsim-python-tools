from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                      os.path.join(
    run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
    transform_from_utm=True)

m9 = catalogue.events_by_number(588, bruce_faults)[0]
m9.plot_slip_evolution(step_size = 5, write="slip_evolution")
