import os

from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.containers.catalogue import RsqSimCatalogue

run_dir = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/bruce/rundir4627"

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir,"zfault_Deepen.in"),
                                                      os.path.join(run_dir,"znames_Deepen.in"),
                                                      transform_from_utm=True)

catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(run_dir, "eqs..out"),
                                                          run_dir, "rundir4627")
