from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.containers.catalogue import RsqSimCatalogue
import os

run_dir = os.path.dirname(__file__)

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                    os.path.join(run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
                                                    transform_from_utm=True)
catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
patches = bruce_faults.find_closest_patches(1626210.185, 5293116.554)
filtered_cat = catalogue.filter_by_patch_numbers(patches)
