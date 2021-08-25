from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import geopandas as gpd
from shapely.geometry import box
from matplotlib import cm

bruce_faults = RsqSimMultiFault.read_fault_file_bruce("../../../../data/bruce/rundir4627/zfault_Deepen.in",
                                                      "../../../../data/bruce/rundir4627/znames_Deepen.in",
                                                      transform_from_utm=True)
# bruce_cat = RsqSimCatalogue.from_catalogue_file_and_lists("../../../../data/bruce/rundir4627/eqs..out",
#                                                           "../../../../data/bruce/rundir4627", "rundir4627")
data = gpd.read_file("dunedin.gpkg")
# dunedin_faults = [fault for fault in bruce_faults.faults if data.intersects(box(*fault.bounds))[0]]
# dunedin_cat = bruce_cat.filter_by_fault(dunedin_faults)
# dunedin_cat.write_csv_and_arrays("dunedin")

dunedin_cat = RsqSimCatalogue.from_csv_and_arrays("dunedin")

#
# test_events = dunedin_cat.events_by_number(dunedin_cat.catalogue_df.index[:100], bruce_faults, min_patches=2)
# background = test_events[0].plot_background(bounds=data.total_bounds, plot_lakes=True,
#                                             plot_highways=True, plot_rivers=True, hillshading_intensity=0.3,
#                                             hillshade_fine=True, pickle_name="temp.pkl", hillshade_cmap=cm.Greys)
# for event in test_events:
#     event.plot_slip_2d(bounds=data.total_bounds, write=f"plots/ev{event.event_id}.png", show=False,
#                        subplots="temp.pkl")