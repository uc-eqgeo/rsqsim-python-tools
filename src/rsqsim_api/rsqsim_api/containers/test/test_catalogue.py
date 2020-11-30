import unittest
import os
import pandas as pd
from rsqsim_api.containers.catalogue import RsqSimCatalogue


class TestReadCatalogue(unittest.TestCase):
    def setUp(self):
        self.catalogue = RsqSimCatalogue()

    def test_from_catalogue_file(self):
        catalogue = self.catalogue.from_catalogue_file(
            os.path.join(os.path.dirname(__file__), 'data/eqs..out'))
        pd.testing.assert_frame_equal(catalogue.catalogue_df,
                                      pd.DataFrame({'t0': [668092957.37445151805877685547, 741329352.47262990474700927734],
                                                    'm0': [12597022496058978.000000, 121238811568268368.000000],
                                                    'mw': [4.666845, 5.322428],
                                                    'x': [192217.149509, 195970.303394],
                                                    'y': [5008104.808942, 5005539.900412],
                                                    'z': [-3200.0, -11700.0],
                                                    'area': [843196.824748, 5283251.259523],
                                                    'dt': [0.497987, 1.352069]}))

    def test_from_csv_and_arrays(self):
        catalogue = self.catalogue.from_csv_and_arrays(
            os.path.join(os.path.dirname(__file__), 'data/bruce_m7/bruce_m7_10kyr'))
        self.assertIsNotNone(catalogue.catalogue_df)
        self.assertIsNotNone(catalogue.event_list)
        self.assertIsNotNone(catalogue.patch_list)
        self.assertIsNotNone(catalogue.patch_slip)
        self.assertIsNotNone(catalogue.patch_time_list)
