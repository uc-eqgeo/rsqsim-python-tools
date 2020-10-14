import os

import pandas as pd

from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.containers.catalogue import RsqSimCatalogue

run_dir = "/home/UOCNT/arh128/PycharmProjects/rnc2/data/examples_EBSZmodel/esc1"

faults = RsqSimMultiFault.from_fault_file_keith(os.path.join(run_dir, "EBSZ_1000.flt"))

catalogue = RsqSimCatalogue.from_catalogue_file_and_lists(os.path.join(run_dir, "eqs.EBSZ_1000_test.out"),
                                                          run_dir, "EBSZ_1000_test")


cat_df = catalogue.catalogue_df


