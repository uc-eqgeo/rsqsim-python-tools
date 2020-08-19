import os

import pandas as pd

from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.containers.catalogue import RsqSimCatalogue

run_dir = "/Users/arh79/PycharmProjects/rnc2/examples_EBSZmodel/esc1"

faults = RsqSimMultiFault.from_fault_file_keith(os.path.join(run_dir, "EBSZ_1000.flt"))

catalogue = RsqSimCatalogue.from_catalogue_file(os.path.join(run_dir, "eqs.EBSZ_1000_test.out"))

cat_df = catalogue.catalogue_array


