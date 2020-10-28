
import os

from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.containers.catalogue import RsqSimCatalogue

run_dir = "/data/bruce/rundir4627"

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir,"zfault_Deepen.in"),
                                                      os.path.join(run_dir,"znames_Deepen.in"),
                                                      transform_from_utm=True)

catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(run_dir, "eqs..out"),
                                                          run_dir, "rundir4627")

m9 = catalogue.events_by_number(588, bruce_faults)[0]
m9.plot_slip_2d()
