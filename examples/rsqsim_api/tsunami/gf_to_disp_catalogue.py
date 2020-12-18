from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(
    run_dir, "../../../data/bruce_m7/bruce_fault_names.in"),
    transform_from_utm=True)

subduction = bruce_faults.filter_by_name("*hik*")
sub_only = catalogue.filter_by_fault(subduction)
events = sub_only.events_by_number(sub_only.catalogue_df.index, bruce_faults)