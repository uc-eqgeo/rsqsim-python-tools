from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.visualisation.animation import AnimateSequence
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                      os.path.join(
    run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
    transform_from_utm=True)

filtered_cat = catalogue.filter_whole_catalogue(max_t0=60e9)
AnimateSequence(filtered_cat, bruce_faults)
