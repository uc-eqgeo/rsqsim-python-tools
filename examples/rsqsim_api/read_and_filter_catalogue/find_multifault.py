from rsqsim_api.containers.catalogue import read_bruce
import os
import shutil

if not all([a in globals() for a in ("bruce_faults", "catalogue")]):
    bruce_faults, catalogue = read_bruce()

if not all([a in globals() for a in ("m75plus", "m75_events")]):
    m75plus = catalogue.filter_df(min_mw=8, max_mw=10)
    m75_events = catalogue.events_by_number(m75plus.index, bruce_faults)

multifault = [ev for ev in m75_events if ev.num_faults > 1]

if os.path.exists("multifault_images"):
    shutil.rmtree("multifault_images")
    os.mkdir("multifault_images")

for i, ev in enumerate(multifault[:100]):
    fname = "multifault_images/mf{:d}.png".format(i)
    ev.plot_slip_2d(show=False, write=fname)



