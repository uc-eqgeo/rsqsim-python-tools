# script to cut long catalogue to a) remove first 60kyr

#import relevant modules
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
import rsqsim_api.io.rsqsim_constants
import os

# Tell python where field paths etc are relative to
script_dir = os.path.abspath('')
fault_dir = "../../../data/shaw2021/rundir5091"
catalogue_dir = fault_dir

# read in catalogue and fault network
fault_model = RsqSimMultiFault.read_fault_file_bruce(os.path.join(script_dir, fault_dir, "zfault_Deepen.in"),
                                                     os.path.join(script_dir, fault_dir, "znames_Deepen.in"),
                                                     transform_from_utm=True)
whole_catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(catalogue_dir, "eqs..out"),\
                                                                list_file_directory=catalogue_dir, list_file_prefix="catalog")


#discard first 60kyr + take next ~200yr
short_cat=whole_catalogue.filter_whole_catalogue(min_t0=2.0e12,max_t0=2.007e12,min_mw=6.0)

# only include events with just one fault
single_evs,single_short_cat=short_cat.find_single_fault(fault_model=fault_model)

# write out
outdir=os.path.join(catalogue_dir,"specfem")
if not os.path.exists(outdir):
    os.mkdir(outdir)

single_short_cat.write_csv_and_arrays(prefix="single_fault_200yr",directory=outdir)