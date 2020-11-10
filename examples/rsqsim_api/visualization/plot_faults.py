from rsqsim_api.containers.fault import read_bruce
from matplotlib import pyplot as plt
import fnmatch

if "bruce_faults" not in globals():
    bruce_faults = read_bruce()
else:
    bruce_faults = globals()["bruce_faults"]

hik = [name for name in bruce_faults.names if fnmatch.fnmatch(name, "hik*")]
puy = ['fiordsz03', 'fiordpusz09']

plt.close("all")
bruce_faults.plot_faults_2d(puy)
