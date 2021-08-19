from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
import os

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(run_dir, "../../../../data/stratos/eqs.10depth_2-5sep_5over_breached.out"),
                                                          os.path.join(run_dir, "../../../../data/stratos"),
                                                          "10depth_2-5sep_5over_breached")

faults = RsqSimMultiFault.read_fault_file_keith(os.path.join(run_dir, "../../../../data/stratos/10depth_2-5sep_5over_breached_segment1.flt"))

events = catalogue.first_n_events(300, faults)
for i, e in enumerate(events):
    e.slip_dist_to_vtk(f"test{i:03}.vtk")


