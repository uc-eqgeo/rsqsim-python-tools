from rsqsim_api.containers.catalogue import read_bruce

if not all([a in globals() for a in ("bruce_faults", "catalogue")]):
    bruce_faults, catalogue = read_bruce()

m9 = catalogue.events_by_number(588, bruce_faults)[0]
m9.plot_slip_2d()
