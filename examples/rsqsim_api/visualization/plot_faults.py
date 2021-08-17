from rsqsim_api.fault.multifault import RsqSimMultiFault
from matplotlib import pyplot as plt
import fnmatch

import os

run_dir = os.path.dirname(__file__)

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                      os.path.join(run_dir, "../../../data/bruce_m7/bruce_names.in"),
                                                      transform_from_utm=True)

hik = [name for name in bruce_faults.names if fnmatch.fnmatch(name, "hik*")]
puy = ['fiordsz03', 'fiordpusz09']

plt.close("all")
bruce_faults.plot_faults_2d(puy)
