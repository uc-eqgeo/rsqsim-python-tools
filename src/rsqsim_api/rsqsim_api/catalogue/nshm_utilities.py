import geopandas as gpd
from shapely.geometry import MultiLineString, LineString
from rsqsim_api.catalogue.utilities import weighted_circular_mean
from rsqsim_api.fault.utilities import calculate_strike
import numpy as np
import os
import pandas as pd


def overall_strike(rup: MultiLineString) -> float:
    """
    Calculate the overall strike of a rupture, which is the average strike of all the segments in the rupture.
    """
    seg_lengths = [seg.length for seg in rup.geoms]
    seg_strikes = [calculate_strike(seg, lt180=True) for seg in rup.geoms]
    seg_strikes = np.array(seg_strikes)
    seg_lengths = np.array(seg_lengths)
    strike = weighted_circular_mean(seg_strikes, seg_lengths)

    return strike

def overall_strike_resolved_length(rup: MultiLineString) -> float:
    """
    Calculate the overall strike of a rupture, which is the average strike of all the segments in the rupture.
    """
    if isinstance(rup, LineString):
        rup = MultiLineString([rup])
    overall_strike_azimuth = overall_strike(rup)
    overall_strike_vector = np.array([np.sin(np.radians(overall_strike_azimuth)), np.cos(np.radians(overall_strike_azimuth))])
    vertices = np.vstack([np.array(seg.coords) for seg in rup.geoms])
    vectors = vertices - np.mean(vertices, axis=0)
    resolved_vectors = np.dot(vectors, overall_strike_vector)
    overall_length = np.max(resolved_vectors) - np.min(resolved_vectors)
    return overall_length


class NshmInversion:
    def __init__(self):
        self.fault_sections = None
        self.rates = None
        self.properties = None
        self.indices = None
        self.num_sections = None
        self.rupture_ids = None
        self.area = None
        self.length = None
        self.mw = None
        self.average_slips = None


    def read_indices(self, indices_csv: str):
        assert self.rupture_ids is not None
        assert os.path.exists(indices_csv)
        with open(indices_csv) as fid:
            data = fid.readlines()
            indices = {}
            num_sections = []
            for line in data[1:]:
                lsplit = np.array([int(index) for index in line.strip().split(",")])

                indices[lsplit[0]] = lsplit[2:]
                num_sections.append(lsplit[1])
        self.indices = indices
        self.num_sections = num_sections

    def read_nshm_inversion(self, directory: str):
        rates_with_zeros = pd.read_csv(os.path.join(directory, "solution/rates.csv"), index_col="Rupture Index")
        self.rupture_ids = np.array(list(rates_with_zeros[rates_with_zeros["Annual Rate"] > 0].index))
        self.rates = rates_with_zeros.loc[self.rupture_ids]["Annual Rate"].values
        self.fault_sections = gpd.read_file(os.path.join(directory, "ruptures/fault_sections.geojson")).to_crs("EPSG:2193")
        self.properties = pd.read_csv(os.path.join(directory, "ruptures/properties.csv"),
                                      index_col="Rupture Index").loc[self.rupture_ids]
        self.area = self.properties["Area (m^2)"].values
        self.length = self.properties["Length (m)"].values
        self.mw = self.properties["Magnitude"].values

        self.read_indices(os.path.join(directory, "ruptures/indices.csv"))
        self.average_slips = pd.read_csv(os.path.join(directory, "ruptures/average_slips.csv"), index_col="Rupture Index").loc[self.rupture_ids]["Average Slip (m)"].values

    def get_rupture_geometry(self, rupture_index: int):
        return self.fault_sections.loc[self.indices[rupture_index]].unary_union

    def ruptures_to_gdf(self):
        rupture_dict = {}
        for index, rate, area, length, mw, average_slip in zip(self.rupture_ids, self.rates, self.area, self.length, self.mw, self.average_slips):
            rupture_dict[index] = {"rate": rate, "area": area, "length": length, "mw": mw, "average_slip": average_slip}
            geom = self.get_rupture_geometry(index)
            rupture_dict[index]["geometry"] = geom
            rupture_dict[index]["resolved_length"] = overall_strike_resolved_length(geom) / 1000

        ruptures = gpd.GeoDataFrame(rupture_dict).T
        ruptures.set_geometry("geometry", inplace=True)
        ruptures.set_crs(epsg=2193, inplace=True)
        return ruptures

    def ruptures_to_geojson(self, filename: str):
        ruptures = self.ruptures_to_gdf()
        ruptures.to_file(filename, driver="GeoJSON")
