from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.tsunami.gf_netcdf import create_lookup_dict, sea_surface_displacements_multi
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(run_dir,
                                                                   "../../../data/bruce_m7/bruce_fault_names.in"),
                                                      transform_from_utm=True)

subduction = bruce_faults.filter_by_name("*hik*")
sub_only = catalogue.filter_by_fault(subduction, minimum_patches_per_fault=10)
events = sub_only.events_by_number(sub_only.catalogue_df.index, bruce_faults)

lookup = create_lookup_dict("bruce_2km_?.nc")

sea_surface_displacements_multi(events, lookup, "test_disp.nc")
