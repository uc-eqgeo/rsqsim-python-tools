"""

"""

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import os

# Get directory where script is run
script_dir = os.path.dirname(__file__)

#
fault_dir = "../../../data/shaw/rundir5091"
catalogue_dir = "../../../data/shaw_m7"


fault_model = RsqSimMultiFault.read_fault_file_bruce(os.path.join(script_dir, fault_dir, "zfault_Deepen.in"),
                                                     os.path.join(script_dir, fault_dir, "znames_Deepen.in"),
                                                     transform_from_utm=True)

catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(script_dir, catalogue_dir, "bruce_m7_10kyr"))
patches = fault_model.find_closest_patches(1626210.185, 5293116.554)
filtered_cat = catalogue.filter_by_patch_numbers(patches)
