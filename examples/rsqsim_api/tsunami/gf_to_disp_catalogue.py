from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.tsunami.gf_netcdf import create_lookup_dict, sea_surface_displacements_multi
import os
"""
Script to create and store (in NetCDF format) sea-surface displacement arrays
from a catalogue of events and a tsunami greens functions database
"""
if __name__ == "__main__":
    run_dir = os.path.dirname(__file__)

    catalogue = RsqSimCatalogue.from_csv_and_arrays(
        os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
    bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.in"),
                                                          os.path.join(run_dir,
                                                                       "../../../data/bruce_m7/bruce_fault_names.in"),
                                                          transform_from_utm=True, multiprocessing=True)

    events = catalogue.events_by_number(catalogue.catalogue_df.index, bruce_faults)

    lookup = create_lookup_dict("bruce_2km_?.nc")

    sea_surface_displacements_multi(events, lookup, "ssd_m7_10kyr.nc")
