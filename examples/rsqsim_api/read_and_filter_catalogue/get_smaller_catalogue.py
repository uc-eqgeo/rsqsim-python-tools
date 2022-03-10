from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import os


# Get directory where script is run
script_dir = os.path.dirname(__file__)

fault_dir = "../../../data/shaw2021/rundir5091"
catalogue_dir = fault_dir

seconds_per_year = 31557600.0

fault_model = RsqSimMultiFault.read_fault_file_bruce(os.path.join(script_dir, fault_dir, "zfault_Deepen.in"),
                                                     os.path.join(script_dir, fault_dir, "znames_Deepen.in"),
                                                     transform_from_utm=True)
whole_catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(catalogue_dir, "eqs..out"),
                                                                list_file_directory=catalogue_dir,
                                                                list_file_prefix="catalog")

short_catalogue = whole_catalogue.filter_whole_catalogue(min_t0=1.e12, max_t0=1.e12 + 2.e4 * seconds_per_year,
                                                         min_mw=7.0)

short_catalogue.write_csv_and_arrays(prefix="shaw_10_kyr_m7_plus", directory="../../../data/shaw_m7")
