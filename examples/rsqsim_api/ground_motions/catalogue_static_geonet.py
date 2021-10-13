from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
import os
import pandas as pd
import h5py
import numpy as np
import pickle

geonet_sites = pd.read_csv("geonet_sm_sites.csv")
site_id, lon, lat = [np.array(a) for a in (geonet_sites.ID, geonet_sites.Longitude, geonet_sites.Latitude)]


run_dir = os.path.dirname(__file__)

catalogue = RsqSimCatalogue.from_csv_and_arrays(
    os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce/rundir4627/zfault_Deepen.in"),
                                                      os.path.join(
    run_dir, "../../../data/bruce/rundir4627/znames_Deepen.in"),
    transform_from_utm=True)

# hdf = h5py.File("geonet_gfs.hdf5", "r")
# displacement_dic = {}
# for i, event in catalogue.catalogue_df.iterrows():
#     print(i)
#     ev_data = catalogue.events_by_number(i, bruce_faults)[0]
#     event_dic = displacement_dic[i] = {"event_id": i, "t0": event.t0, "m0": event.m0, "mw": event.mw,
#                                        "lon": lon, "lat": lat, "site_id": site_id}
#     x, y, z = [np.zeros(lon.shape) for i in range(3)]
#     for patch_i, patch_slip in zip(ev_data.patch_numbers, ev_data.patch_slip):
#         x += patch_slip * hdf[str(patch_i)]["x"]
#         y += patch_slip * hdf[str(patch_i)]["y"]
#         z += patch_slip * hdf[str(patch_i)]["z"]
#     event_dic["x"] = x
#     event_dic["y"] = y
#     event_dic["z"] = z
#
# hdf.close()

displacement_dic = pickle.load(open("bruce_m7_10kyr_static_geonet.pkl", "rb"))

catalogue.catalogue_df.to_csv("geonet_static_displacements/all_events_summary.csv")
for event_id, event in displacement_dic.items():
    trimmed_ev = {key: value for key, value in event.items() if key in ["lon", "lat", "site_id", "x", "y", "z"]}
    ev_df = pd.DataFrame(trimmed_ev)
    ev_df.to_csv("geonet_static_displacements/events{:d}.csv".format(event_id))



