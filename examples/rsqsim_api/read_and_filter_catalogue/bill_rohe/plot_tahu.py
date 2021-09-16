from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import geopandas as gpd
from shapely.geometry import box
from matplotlib import cm

bruce_faults = RsqSimMultiFault.read_fault_file_bruce("../../../../data/bruce/rundir4627/zfault_Deepen.in",
                                                      "../../../../data/bruce/rundir4627/znames_Deepen.in",
                                                      transform_from_utm=True)
tahu_cat = RsqSimCatalogue.from_csv_and_arrays("tahu_rohe")
data = gpd.read_file("ngati_tahu.gpkg")

test_events = tahu_cat.events_by_number(tahu_cat.catalogue_df.index[:100], bruce_faults, min_patches=2)
background = test_events[0].plot_background(bounds=data.total_bounds, plot_lakes=True,
                                            plot_highways=True, plot_rivers=True, hillshading_intensity=0.3,
                                            hillshade_fine=True, pickle_name="temp.pkl", hillshade_cmap=cm.Greys)
for event in test_events:
    event.plot_slip_2d(bounds=data.total_bounds, write=f"plots/ev{event.event_id}.png", show=False,
                       subplots="temp.pkl")
    event.slip_dist_to_txt(f"slip_dists/ev{event.event_id}_no_zeros.txt", nztm_to_lonlat=True, include_zeros=False)
    event.slip_dist_to_txt(f"slip_dists/ev{event.event_id}_inc_zeros.txt", nztm_to_lonlat=True)