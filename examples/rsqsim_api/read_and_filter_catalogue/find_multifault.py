from rsqsim_api.fault.multifault import RsqSimMultiFault
import shutil
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import os

run_dir = os.path.dirname(__file__)

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../data/bruce_m7/bruce_names.in"),
                                                      transform_from_utm=True)
catalogue = RsqSimCatalogue.from_csv_and_arrays(os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))

m75plus = catalogue.filter_df(min_mw=8)
m75_events = catalogue.events_by_number(m75plus.index, bruce_faults, min_patches=50)

multifault = [ev for ev in m75_events if ev.num_faults > 1]

if os.path.exists("multifault_images"):
    shutil.rmtree("multifault_images")

os.mkdir("multifault_images")

for i, ev in enumerate(multifault[:100]):
    fname = "multifault_images/mf{:d}.png".format(i)
    ev.plot_slip_2d(show=False, write=fname)



