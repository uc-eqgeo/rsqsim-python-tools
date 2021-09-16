from rsqsim_api.fault.multifault import RsqSimMultiFault
import shutil
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import os

run_dir = os.path.dirname(__file__)

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../data/bruce_m7/bruce_names.in"),
                                                      transform_from_utm=True)
catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))

m75_events = catalogue.events_by_number(catalogue.catalogue_df.index, bruce_faults, min_patches=50)

multifault = [ev for ev in m75_events if ev.num_faults > 1]

if os.path.exists("multifault_images"):
    shutil.rmtree("multifault_images")

os.mkdir("multifault_images")

for i, ev in enumerate(multifault[:100]):
    fname = f"multifault_images/event{ev.event_id}_without_zero_slip.png"
    ev.plot_slip_2d(show=False, write=fname, plot_zeros=False)
    fname = f"multifault_images/event{ev.event_id}_with_whole_faults.png"
    ev.plot_slip_2d(show=False, write=fname)



