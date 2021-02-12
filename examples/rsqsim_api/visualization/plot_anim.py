from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.animation import AnimateSequence
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                      os.path.join(
    run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
    transform_from_utm=True)

filtered_cat = catalogue.filter_whole_catalogue(
    min_t0=1000*3.154e7, max_t0=2000*3.154e7)  # 1000 years
AnimateSequence(filtered_cat, bruce_faults, write="demo")
AnimateSequence(filtered_cat, bruce_faults, write="demoHillshade", hillshading_intensity=0.5)
AnimateSequence(filtered_cat, bruce_faults, write="demoMovie", file_format="mp4", figsize=(10, 8))
