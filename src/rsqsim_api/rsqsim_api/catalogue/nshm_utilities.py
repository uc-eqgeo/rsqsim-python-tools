"""
Utilities for reading and processing New Zealand NSHM inversion results.

Provides functions for computing overall rupture strike and resolved
length from multi-segment traces, and the :class:`NshmInversion` class
for loading OpenSHA fault-system solution directories.
"""
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString
from rsqsim_api.catalogue.utilities import weighted_circular_mean
from rsqsim_api.fault.utilities import calculate_strike
import numpy as np
import os
import pandas as pd


def overall_strike(rup: MultiLineString) -> float:
    """
    Compute the length-weighted mean strike of a multi-segment rupture.

    Parameters
    ----------
    rup : shapely.geometry.MultiLineString
        Rupture trace composed of one or more line segments.

    Returns
    -------
    float
        Weighted circular mean strike in degrees (0–180 convention as
        returned by :func:`~rsqsim_api.fault.utilities.calculate_strike`).
    """
    seg_lengths = [seg.length for seg in rup.geoms]
    seg_strikes = [calculate_strike(seg, lt180=True) for seg in rup.geoms]
    seg_strikes = np.array(seg_strikes)
    seg_lengths = np.array(seg_lengths)
    strike = weighted_circular_mean(seg_strikes, seg_lengths)

    return strike

def overall_strike_resolved_length(rup: MultiLineString) -> float:
    """
    Compute the resolved length of a rupture along its mean strike direction.

    Projects all rupture vertices onto the overall strike vector and
    returns the range (max minus min) of those projections.

    Parameters
    ----------
    rup : shapely.geometry.MultiLineString or LineString
        Rupture trace.  A single ``LineString`` is automatically wrapped
        in a ``MultiLineString``.

    Returns
    -------
    float
        Resolved rupture length in the same units as the input geometry
        (metres for NZTM).
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
    """
    Reader for an OpenSHA fault-system inversion solution directory.

    Loads fault sections, rupture rates, properties, connectivity
    indices, and average slips from an NSHM inversion output directory
    in the standard OpenSHA CSV/GeoJSON layout.

    Attributes
    ----------
    fault_sections : geopandas.GeoDataFrame or None
        GeoDataFrame of fault-section geometries (EPSG:2193).
    rates : numpy.ndarray or None
        Annual occurrence rates for ruptures with rate > 0.
    properties : pandas.DataFrame or None
        Rupture properties (area, length, magnitude) for non-zero-rate
        ruptures.
    indices : dict or None
        Mapping of rupture ID to array of participating fault-section
        indices.
    num_sections : list or None
        Number of sections per rupture.
    rupture_ids : numpy.ndarray or None
        Indices of ruptures with non-zero annual rate.
    area : numpy.ndarray or None
        Rupture areas in m².
    length : numpy.ndarray or None
        Rupture lengths in m.
    mw : numpy.ndarray or None
        Moment magnitudes.
    average_slips : numpy.ndarray or None
        Average slip per rupture in m.
    """

    def __init__(self):
        """Initialise an empty NshmInversion; call :meth:`read_nshm_inversion` to populate."""
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
        """
        Read the rupture–section connectivity index file.

        Parses a CSV where each row lists a rupture ID, the number of
        sections, and the section indices.  Populates :attr:`indices`
        and :attr:`num_sections`.

        Parameters
        ----------
        indices_csv : str
            Path to ``ruptures/indices.csv`` inside the solution
            directory.

        Raises
        ------
        AssertionError
            If :attr:`rupture_ids` has not been set, or if the file
            does not exist.
        """
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
        """
        Load an NSHM inversion solution from a directory.

        Reads rates, fault sections, properties, connectivity indices,
        and average slips.  Only ruptures with a positive annual rate
        are retained.

        Parameters
        ----------
        directory : str
            Path to the inversion solution directory, which must contain:
            ``solution/rates.csv``,
            ``ruptures/fault_sections.geojson``,
            ``ruptures/properties.csv``,
            ``ruptures/indices.csv``, and
            ``ruptures/average_slips.csv``.
        """
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
        """
        Return the unary union geometry of all sections for a rupture.

        Parameters
        ----------
        rupture_index : int
            Rupture ID from :attr:`rupture_ids`.

        Returns
        -------
        shapely.geometry.base.BaseGeometry
            Union of all fault-section geometries participating in the
            specified rupture.
        """
        return self.fault_sections.loc[self.indices[rupture_index]].unary_union

    def ruptures_to_gdf(self):
        """
        Build a GeoDataFrame of all non-zero-rate ruptures.

        For each rupture, computes the unary-union geometry of its
        sections, the resolved length along the mean strike, and
        assembles rate, area, length, magnitude, and average slip.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame (EPSG:2193) with one row per rupture and
            columns ``["rate", "area", "length", "mw", "average_slip",
            "geometry", "resolved_length"]``.  ``resolved_length`` is
            in km.
        """
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
        """
        Write all non-zero-rate ruptures to a GeoJSON file.

        Parameters
        ----------
        filename : str
            Output GeoJSON file path.
        """
        ruptures = self.ruptures_to_gdf()
        ruptures.to_file(filename, driver="GeoJSON")
