from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
import os
import pandas as pd
import numpy as np
import time
import h5py

geonet_sites = pd.read_csv("geonet_sm_sites.csv")

run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce/rundir4627/zfault_Deepen.in"),
                                                      os.path.join(
    run_dir, "../../../data/bruce/rundir4627/znames_Deepen.in"),
    transform_from_utm=True)


x, y = np.array(geonet_sites.nztm_x), np.array(geonet_sites.nztm_y)
z = np.zeros(x.shape)

start = time.time()
hdf = h5py.File("geonet_gfs.hdf5", "w")

for i, patch in bruce_faults.patch_dic.items():
    gfs = patch.calculate_3d_greens_functions(x, y, z)
    group = hdf.create_group(str(i))
    for component in ("x", "y", "z"):
        group.create_dataset(component, data=gfs[component])

hdf.close()

end = time.time()

print(end - start)
