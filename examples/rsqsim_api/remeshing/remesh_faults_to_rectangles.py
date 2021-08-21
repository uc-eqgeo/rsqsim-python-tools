from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../data/bruce_m7/bruce_names.in"),
                                                      transform_from_utm=True)

events = catalogue.first_n_events(50, bruce_faults)

if not os.path.exists("meshes_sd"):
    os.mkdir("meshes_sd")

for i, event in enumerate(events):
    event.slip_dist_to_obj(f"meshes_sd/event{i}.obj", include_zeros=False)

# for fault in bruce_faults.faults:
#     stl_name = f"meshes/{fault.name}.stl"
#     fault.to_stl(stl_name)
