"""
RSQSim earthquake event representation.

Provides :class:`RsqSimEvent`, which stores the catalogue parameters
(time, magnitude, location) and the per-patch slip distribution for a
single simulated rupture, together with methods for computing derived
quantities and exporting slip distributions in a variety of formats.

Also provides :class:`OpenQuakeMultiSquareRupture` for packaging events
as multi-planar ruptures compatible with the OpenQuake engine.
"""
import fnmatch
import json
import operator
import os
import pickle
import xml.etree.ElementTree as ElemTree
from collections import defaultdict
from collections.abc import Iterable
from math import isclose
from string import digits
from xml.dom import minidom

import geopandas as gpd
import meshio
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Transformer
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from scipy.spatial import KDTree

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.fault.patch import OpenQuakeRectangularPatch
from rsqsim_api.io.bruce_shaw_utilities import bruce_subduction
from rsqsim_api.io.mesh_utils import array_to_mesh, quads_to_vtk
from rsqsim_api.visualisation.utilities import plot_coast, plot_background
from rsqsim_api.catalogue.utilities import weighted_circular_mean, m0_to_mw

transformer_nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)


class RsqSimEvent:
    """
    A single earthquake event from an RSQSim catalogue.

    Stores basic seismic catalogue parameters and, when populated,
    the patch-level slip distribution including patch objects, slip
    magnitudes, and rupture timing.

    Attributes
    ----------
    event_id : int or None
        Unique integer identifier for the event in the catalogue.
    t0 : float or None
        Origin time in seconds from the start of the simulation.
    m0 : float or None
        Scalar seismic moment in N·m.
    mw : float or None
        Moment magnitude.
    x, y, z : float or None
        Hypocentre coordinates in NZTM (metres, EPSG:2193).  ``z`` is
        negative downward.
    area : float or None
        Total rupture area in m².
    dt : float or None
        Rupture duration in seconds.
    patches : list or None
        List of :class:`~rsqsim_api.fault.patch.RsqSimTriangularPatch`
        objects that slipped in this event.
    patch_slip : numpy.ndarray or None
        Slip magnitude (m) for each entry in ``patches``.
    faults : list or None
        Unique :class:`~rsqsim_api.fault.segment.RsqSimSegment` objects
        involved in this event.
    patch_time : numpy.ndarray or None
        Absolute rupture time (s) for each patch.
    patch_numbers : numpy.ndarray or None
        Global patch IDs corresponding to ``patches``.
    length : float or None
        Estimated rupture length (m) along fault traces.
    """

    def __init__(self):
        """
        Initialise an empty RsqSimEvent.

        All attributes are set to ``None``.  Use the class methods
        :meth:`from_catalogue_array` or :meth:`from_earthquake_list` to
        construct a populated event.
        """
        # Event ID
        self.event_id = None
        # Origin time
        self.t0 = None
        # Seismic moment and mw
        self.m0 = None
        self.mw = None
        # Hypocentre location
        self.x, self.y, self.z = (None,) * 3
        # Rupture area
        self.area = None
        # Rupture duration
        self.dt = None

        # Parameters for slip distributions
        self.patches = None
        self.patch_slip = None
        self.faults = None
        self.patch_time = None
        self.patch_numbers = None
        self._mean_slip = None
        self.length = None
        self._mean_strike = None
        self._mean_strike_180 = None
        self._mean_dip = None
        self._mean_rake = None
        self._first_fault = None

    @property
    def num_faults(self):
        """Number of unique fault segments involved in this event."""
        return len(self.faults)

    @property
    def bounds(self):
        """
        Axis-aligned bounding box of the rupture in NZTM coordinates.

        Returns
        -------
        list of float
            ``[x_min, y_min, x_max, y_max]`` in metres (NZTM).
        """
        x1 = min([min(fault.vertices[:, 0]) for fault in self.faults])
        y1 = min([min(fault.vertices[:, 1]) for fault in self.faults])
        x2 = max([max(fault.vertices[:, 0]) for fault in self.faults])
        y2 = max([max(fault.vertices[:, 1]) for fault in self.faults])
        return [x1, y1, x2, y2]

    @property
    def exterior(self):
        """
        Shapely geometry representing the union of all ruptured patch polygons.

        Returns
        -------
        shapely.geometry.base.BaseGeometry
            Union of all patch outlines as a Shapely geometry object.
        """
        return unary_union([patch.as_polygon() for patch in self.patches])

    @property
    def mean_slip(self):
        """Mean slip (m) averaged over all ruptured patches."""
        return self._mean_slip

    @property
    def mean_strike(self):
        """Mean strike (degrees, 0–360) averaged over all ruptured patches."""
        return self._mean_strike

    @property
    def mean_strike_180(self):
        """Mean strike (degrees, 0–180) averaged over all ruptured patches."""
        return self._mean_strike_180

    @property
    def mean_dip(self):
        """Mean dip (degrees) averaged over all ruptured patches."""
        return self._mean_dip

    @property
    def mean_rake(self):
        """Mean rake (degrees) averaged over all ruptured patches."""
        return self._mean_rake

    def find_first_fault(self, fault_model: RsqSimMultiFault, name: bool = True):
        """
        Return the fault that first ruptured in this event.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model used to look up patch-to-fault associations.
        name : bool, optional
            If ``True`` (default), return the fault name string.
            If ``False``, return the fault segment object.

        Returns
        -------
        str or RsqSimSegment
            The name (or object) of the fault that hosted the first
            ruptured patch.
        """
        first_patch = self.patches[np.where(self.patch_time == np.min(self.patch_time))[0][0]]
        first_fault = fault_model.faults_with_patches[first_patch.patch_number]
        if name:
            return first_fault.name
        else:
            return first_fault

    @classmethod
    def from_catalogue_array(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float, event_id: int = None):
        """
        Construct an event from basic catalogue parameters.

        Creates an event with catalogue-level metadata only (no patch
        slip distribution).

        Parameters
        ----------
        t0 : float
            Origin time in seconds from the simulation start.
        m0 : float
            Scalar seismic moment in N·m.
        mw : float
            Moment magnitude.
        x : float
            Hypocentre easting in NZTM (m).
        y : float
            Hypocentre northing in NZTM (m).
        z : float
            Hypocentre depth in NZTM (m, negative downward).
        area : float
            Total rupture area in m².
        dt : float
            Rupture duration in seconds.
        event_id : int, optional
            Unique catalogue event ID.

        Returns
        -------
        RsqSimEvent
            Event populated with catalogue parameters; patch attributes
            remain ``None``.
        """

        event = cls()
        event.t0, event.m0, event.mw, event.x, event.y, event.z = [t0, m0, mw, x, y, z]
        event.area, event.dt = [area, dt]
        event.event_id = event_id

        return event

    @property
    def patch_outline_gs(self):
        """
        GeoSeries of individual patch outlines in NZTM (EPSG:2193).

        Returns
        -------
        geopandas.GeoSeries
            One Shapely polygon per ruptured patch, with CRS set to
            EPSG:2193 (NZTM).
        """
        return gpd.GeoSeries([patch.as_polygon() for patch in self.patches], crs=2193)

    @classmethod
    def from_earthquake_list(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: list | np.ndarray | tuple,
                             patch_slip: list | np.ndarray | tuple,
                             patch_time: list | np.ndarray | tuple,
                             fault_model: RsqSimMultiFault, filter_single_patches: bool = True,
                             min_patches: int = 10, min_slip: float | int = 1, event_id: int = None):
        """
        Construct an event from catalogue parameters and patch slip data.

        Filters out fault segments that contribute fewer than
        ``min_patches`` patches, then resolves patch IDs to patch objects
        and fault objects using ``fault_model``.

        Parameters
        ----------
        t0 : float
            Origin time in seconds.
        m0 : float
            Scalar seismic moment in N·m.
        mw : float
            Moment magnitude.
        x : float
            Hypocentre easting in NZTM (m).
        y : float
            Hypocentre northing in NZTM (m).
        z : float
            Hypocentre depth in NZTM (m, negative downward).
        area : float
            Total rupture area in m².
        dt : float
            Rupture duration in seconds.
        patch_numbers : array-like
            Global patch IDs that slipped in this event.
        patch_slip : array-like
            Slip magnitude (m) for each entry in ``patch_numbers``.
        patch_time : array-like
            Absolute rupture time (s) for each entry in ``patch_numbers``.
        fault_model : RsqSimMultiFault
            Fault model providing the patch and fault dictionaries.
        filter_single_patches : bool, optional
            Unused; retained for API compatibility.
        min_patches : int, optional
            Minimum number of patches a fault must contribute to be
            retained.  Defaults to 10.
        min_slip : float or int, optional
            Unused; retained for API compatibility.
        event_id : int, optional
            Unique catalogue event ID.

        Returns
        -------
        RsqSimEvent
            Fully populated event including patch slip distribution and
            fault objects.
        """
        event = cls.from_catalogue_array(
            t0, m0, mw, x, y, z, area, dt, event_id=event_id)

        faults_with_patches = fault_model.faults_with_patches
        patches_on_fault = defaultdict(list)
        [patches_on_fault[faults_with_patches[i]].append(i) for i in patch_numbers]

        mask = np.full(len(patch_numbers), True)
        for fault in patches_on_fault.keys():
            patches_on_this_fault = patches_on_fault[fault]
            if len(patches_on_this_fault) < min_patches:
                # Finds the indices of values in patches_on_this_fault in patch_numbers
                patch_on_fault_indices = np.searchsorted(patch_numbers, patches_on_this_fault)
                mask[patch_on_fault_indices] = False

        event.patch_numbers = patch_numbers[mask]
        event.patch_slip = patch_slip[mask]
        event.patch_time = patch_time[mask]

        if event.patch_numbers.size > 1:
            patchnum_lookup = operator.itemgetter(*(event.patch_numbers))
            event.patches = list(patchnum_lookup(fault_model.patch_dic))
            event.faults = list(set(patchnum_lookup(fault_model.faults_with_patches)))

        elif event.patch_numbers.size == 1:
            event.patches = [fault_model.patch_dic[event.patch_numbers[0]]]
            event.faults = [fault_model.faults_with_patches[event.patch_numbers[0]]]

        else:
            event.patches = []
            event.faults = []
            print(
                f"Event {event_id} doesn't rupture more than {min_patches} patches on any fault. \n Decrease min_patches if you want a fault + patches returned.")

        return event

    @classmethod
    def from_multiprocessing(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: list | np.ndarray | tuple,
                             patch_slip: list | np.ndarray | tuple,
                             patch_time: list | np.ndarray | tuple,
                             fault_model: RsqSimMultiFault, mask: list, event_id: int = None):
        """
        Construct an event from pre-computed multiprocessing results.

        Similar to :meth:`from_earthquake_list`, but accepts a boolean
        ``mask`` that has already been applied to filter patches, rather
        than recomputing it here.

        Parameters
        ----------
        t0 : float
            Origin time in seconds.
        m0 : float
            Scalar seismic moment in N·m.
        mw : float
            Moment magnitude.
        x : float
            Hypocentre easting in NZTM (m).
        y : float
            Hypocentre northing in NZTM (m).
        z : float
            Hypocentre depth in NZTM (m, negative downward).
        area : float
            Total rupture area in m².
        dt : float
            Rupture duration in seconds.
        patch_numbers : array-like
            Global patch IDs that slipped in this event.
        patch_slip : array-like
            Slip magnitude (m) for each entry in ``patch_numbers``.
        patch_time : array-like
            Absolute rupture time (s) for each entry in ``patch_numbers``.
        fault_model : RsqSimMultiFault
            Fault model providing patch and fault dictionaries.
        mask : list of bool
            Boolean mask selecting the patches to retain from
            ``patch_numbers``.
        event_id : int, optional
            Unique catalogue event ID.

        Returns
        -------
        RsqSimEvent
            Fully populated event.
        """
        event = cls.from_catalogue_array(
            t0, m0, mw, x, y, z, area, dt, event_id=event_id)
        event.patch_numbers = patch_numbers[mask]
        event.patch_slip = patch_slip[mask]
        event.patch_time = patch_time[mask]

        if event.patch_numbers.size > 0:
            patchnum_lookup = operator.itemgetter(*(event.patch_numbers))
            event.patches = list(patchnum_lookup(fault_model.patch_dic))
            event.faults = list(set(patchnum_lookup(fault_model.faults_with_patches)))

        else:
            event.patches = []
            event.faults = []

        return event
    
    def sub_events_by_fault(self, fault_model: RsqSimMultiFault, min_slip: float = 0.1) -> list["RsqSimEvent"]:
        """
        Split the event into per-fault sub-events.

        For each fault involved in the event, a new :class:`RsqSimEvent`
        is created containing only the patches on that fault.  Faults
        where no patch has slip >= ``min_slip`` are skipped.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch and fault dictionaries.
        min_slip : float, optional
            Minimum slip threshold (m) for a patch to be included in a
            sub-event.  Defaults to 0.1 m.

        Returns
        -------
        list of RsqSimEvent
            One event per fault that contributed qualifying slip.
        """
        m0s = self.make_fault_moment_dict(fault_model=fault_model, mu=3.0e10, by_cfm_names=False)
        subevents = []
        for fault in self.faults:
            fault_patches = np.array(list(fault.patch_dic.keys()))
            fault_patch_numbers = self.patch_numbers[np.isin(self.patch_numbers, fault_patches)]
            fault_patch_slip = self.patch_slip[np.isin(self.patch_numbers, fault_patch_numbers)]
            fault_patch_time = self.patch_time[np.isin(self.patch_numbers, fault_patch_numbers)]

            fault_t0 = self.t0 + np.min(fault_patch_time)
            fault_dt = np.max(fault_patch_time) - np.min(fault_patch_time)
            fault_m0 = m0s[fault.name]
            fault_mw = m0_to_mw(fault_m0)
            fault_area = np.sum([fault.patch_dic[number].area for number in fault_patch_numbers])
            fault_first_patch = fault_patch_numbers[np.where(fault_patch_time == np.min(fault_patch_time))[0][0]]
            fault_first_patch = fault_model.patch_dic[fault_first_patch]
            fault_x, fault_y, fault_z = fault_first_patch.centre



            # Check if any slip is above the minimum slip threshold
            if np.any(fault_patch_slip >= min_slip):
                subevent = RsqSimEvent.from_earthquake_list(
                    fault_t0, fault_m0, fault_mw, fault_x, fault_y, fault_z, fault_area, fault_dt,
                    fault_patch_numbers, fault_patch_slip, fault_patch_time,
                    fault_model=fault_model, event_id=self.event_id)
                subevents.append(subevent)

        return subevents


    def find_mean_slip(self):
        """
        Compute and cache the mean slip across all ruptured patches.

        Sets :attr:`_mean_slip` (accessed via :attr:`mean_slip`) to the
        arithmetic mean of ``patch_slip``.  Does nothing if no patches
        are loaded.
        """
        if self.patches:
            total_slip = np.sum(self.patch_slip)
            npatches = len(self.patches)
            if all([total_slip > 0., npatches > 0]):
                self._mean_slip = total_slip / npatches

    def find_mean_strike(self):
        """
        Compute and cache the arithmetic mean strike (0–360°).

        Sets :attr:`_mean_strike` (accessed via :attr:`mean_strike`).
        Does nothing if no patches are loaded.
        """
        if self.patches:
            cumstrike = 0.
            for patch in self.patches:
                cumstrike += patch.strike
            npatches = len(self.patches)
            if npatches > 0:
                self._mean_strike = cumstrike / npatches

    def find_mean_strike_180(self):
        """
        Compute and cache the mean strike folded into the range 0–180°.

        Strikes in 180–360° are reduced by 180° before averaging, so
        that conjugate orientations are treated as equivalent.  Sets
        :attr:`_mean_strike_180` (accessed via :attr:`mean_strike_180`).
        Does nothing if no patches are loaded.
        """
        if self.patches:
            cumstrike = 0.
            for patch in self.patches:
                strike = patch.strike
                if 0.<= strike <180:
                    cumstrike += strike
                else:
                    assert(0.<= strike - 180 < 180), "strike not in range 0 - 360"
                    cumstrike += (strike - 180.)
            npatches = len(self.patches)
            if npatches > 0:
                self._mean_strike_180 = cumstrike / npatches

    def find_mean_dip(self):
        """
        Compute and cache the arithmetic mean dip (degrees).

        Sets :attr:`_mean_dip` (accessed via :attr:`mean_dip`).
        Does nothing if no patches are loaded.
        """
        if self.patches:
            cumdip = 0.
            npatches = len(self.patches)
            for patch in self.patches:
                cumdip += patch.dip

            if npatches > 0:
                self._mean_dip = cumdip / npatches

    def find_mean_rake(self):
        """
        Compute and cache the arithmetic mean rake (degrees).

        Sets :attr:`_mean_rake` (accessed via :attr:`mean_rake`).
        Does nothing if no patches are loaded.
        """
        if self.patches:
            cumrake = 0.
            for patch in self.patches:
                cumrake += patch.rake
            npatches = len(self.patches)
            if npatches > 0:
                self._mean_rake = cumrake / npatches

    def find_length(self, min_slip_percentile: float | None = None):
        """
        Estimate and cache the rupture length along fault surface traces.

        For each fault, the range of distances of patch centroids
        projected onto the fault trace is summed to give a cumulative
        rupture length.  Sets :attr:`length`.

        Parameters
        ----------
        min_slip_percentile : float or None, optional
            Unused; reserved for future filtering by slip percentile.
        """
        if self.patches:
            rupture_length = 0.
            for fault in self.faults:
                fault_trace = fault.trace
                patch_locs = []
                for patch in self.patches:
                    centroid = Point(patch.centre[:2])

                    patch_dist = fault_trace.project(centroid)
                    patch_locs.append(patch_dist)
                rupture_length += np.ptp(patch_locs)

        self.length = rupture_length

    def make_fault_moment_dict(self, fault_model: RsqSimMultiFault, mu: float = 3.0e10, by_cfm_names: bool = True,
                               min_m0: float = 0.):
        """
        Build a dictionary of seismic moment released on each fault.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch and fault dictionaries.
        mu : float, optional
            Shear modulus in Pa.  Defaults to 3×10¹⁰ Pa (30 GPa).
        by_cfm_names : bool, optional
            If ``True`` (default), merge sub-segments by stripping
            trailing digits and hyphens to recover CFM fault names.
            If ``False``, use the raw fault segment names.
        min_m0 : float, optional
            Minimum moment (N·m) for a fault to be included.  Defaults
            to 0 (include all faults).

        Returns
        -------
        dict
            Mapping of fault name (str) to scalar moment (float, N·m).
        """
        assert self.faults is not None, "Event has no faults, can't calculate moment"
        m0_dict = {}
        for fault in self.faults:
            fault_patches = np.array(list(fault.patch_dic.keys()))
            fault_patch_numbers = self.patch_numbers[np.isin(self.patch_numbers, fault_patches)]
            fault_patch_slip = self.patch_slip[np.isin(self.patch_numbers, fault_patches)]
            areas = [fault.patch_dic[number].area for number in fault_patch_numbers]
            m0 = sum(fault_patch_slip * areas * mu)
            if m0 > min_m0:
                m0_dict[fault.name] = m0

        if by_cfm_names:
            # make lookup for fault segment names
            fault_short_dict = {}
            fault_short_names = []
            rm_digs = str.maketrans('', '', digits)
            rm_hyph = str.maketrans('', '', "-")
            for name in fault_model.names:
                short_name = name.translate(rm_digs)
                short_name = short_name.translate(rm_hyph)
                fault_short_names += short_name
                fault_short_dict[name] = short_name

            m0_df = pd.DataFrame.from_dict(m0_dict, orient='index', columns=['M0'])
            m0_df["segName"] = [fault_short_dict[key] for key in m0_df.index]
            m0_df.reset_index(inplace=True, drop=True)
            m0_per_segment = m0_df.groupby(by="segName").sum()
            m0_seg_dict = dict(zip(m0_per_segment.index, m0_per_segment.M0))
        else:
            m0_seg_dict = m0_dict

        return m0_seg_dict

    def make_moment_prop_dict(self, fault_model: RsqSimMultiFault, mu: float = 3.0e10, by_cfm_names: bool = True):
        """
        Build a dictionary of each fault's fractional contribution to total moment.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch and fault dictionaries.
        mu : float, optional
            Shear modulus in Pa.  Defaults to 3×10¹⁰ Pa.
        by_cfm_names : bool, optional
            Passed to :meth:`make_fault_moment_dict`; if ``True``
            (default), merge sub-segments to CFM fault names.

        Returns
        -------
        dict
            Mapping of fault name (str) to proportion of total event
            moment (float, 0–1), sorted in descending order.
        """
        m0_dict = self.make_fault_moment_dict(fault_model=fault_model, mu=mu, by_cfm_names=by_cfm_names)

        prop_dict = {}
        for fault_name in m0_dict.keys():
            prop_dict[fault_name] = m0_dict[fault_name] / self.m0

        # and sort
        prop_dict_sorted_keys = sorted(prop_dict, key=prop_dict.get, reverse=True)
        prop_dict_sorted = dict(zip(prop_dict_sorted_keys, [prop_dict[x] for x in prop_dict_sorted_keys]))

        return prop_dict_sorted

    def plot_slip_2d(self, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", show: bool = True,
                     extra_sub_list: list = None,
                     write: str = None, subplots=None, global_max_sub_slip: int = 0, global_max_slip: int = 0,
                     figsize: tuple = (6.4, 4.8), hillshading_intensity: float = 0.0, bounds: tuple = None,
                     plot_rivers: bool = True, plot_lakes: bool = True,
                     plot_highways: bool = True, plot_boundaries: bool = False, create_background: bool = False,
                     coast_only: bool = True, hillshade_cmap: colors.LinearSegmentedColormap = cm.terrain,
                     plot_log_scale: bool = False, log_cmap: str = "magma", log_min: float = 1.0,
                     log_max: float = 100., plot_traces: bool = True, trace_colour: str = "pink",
                     land_color: str = 'antiquewhite',
                     min_slip_percentile: float = None, min_slip_value: float = None, plot_zeros: bool = True,
                     wgs: bool = False, title: str = None,
                     plot_edge_label: bool = True, plot_cbars: bool = True, alpha: float = 1.0,
                     coast_on_top: bool = False):
        """
        Plot a 2-D map of the slip distribution for this event.

        Subduction-interface faults and crustal faults are coloured with
        separate colourmaps.  Optionally overlays a coastal/background
        map and saves to file.

        Parameters
        ----------
        subduction_cmap : str, optional
            Matplotlib colourmap name for subduction-interface patches.
            Defaults to ``"plasma"``.
        crustal_cmap : str, optional
            Matplotlib colourmap name for crustal patches.  Defaults to
            ``"viridis"``.
        show : bool, optional
            If ``True`` (default), call ``plt.show()`` after plotting.
        extra_sub_list : list of str, optional
            Additional fault names to treat as subduction interface,
            supplementing the built-in ``bruce_subduction`` list.
        write : str or None, optional
            File path to save the figure (e.g. ``"event_001.png"``).
            If ``None``, the figure is not saved.
        subplots : tuple or str or None, optional
            Existing ``(fig, ax)`` tuple or path to a pickled figure to
            plot on top of.  If ``None``, a new figure is created.
        global_max_sub_slip : float, optional
            Fixed colourbar maximum for subduction slip.  If 0, the
            per-event maximum is used.
        global_max_slip : float, optional
            Fixed colourbar maximum for crustal slip.  If 0, the
            per-event maximum is used.
        figsize : tuple of float, optional
            Figure size in inches ``(width, height)``.
        hillshading_intensity : float, optional
            Hillshading intensity passed to ``plot_background``.
        bounds : tuple or None, optional
            Map extent ``(x_min, y_min, x_max, y_max)`` in NZTM.
            Defaults to the event bounding box.
        plot_rivers, plot_lakes, plot_highways, plot_boundaries : bool, optional
            Toggle background map layers.
        create_background : bool, optional
            If ``True``, render a full background map.
        coast_only : bool, optional
            If ``True`` (default), only render the coastline as background.
        hillshade_cmap : LinearSegmentedColormap, optional
            Colourmap for hillshading.
        plot_log_scale : bool, optional
            If ``True``, use logarithmic colour scaling.
        log_cmap : str, optional
            Colourmap for log-scale plots.  Defaults to ``"magma"``.
        log_min, log_max : float, optional
            Colour scale limits for log-scale plots.
        plot_traces : bool, optional
            If ``True``, plot fault surface traces.
        trace_colour : str, optional
            Colour for fault traces.
        land_color : str, optional
            Background land colour.
        min_slip_percentile : float or None, optional
            If set, only show patches above this slip percentile.
        min_slip_value : float or None, optional
            If set, only show patches with slip >= this value (m).
        plot_zeros : bool, optional
            If ``True`` (default), plot zero-slip patches.
        wgs : bool, optional
            If ``True``, use WGS84 coordinates rather than NZTM.
        title : str or None, optional
            Figure super-title.
        plot_edge_label : bool, optional
            If ``True``, show axis edge labels on the background map.
        plot_cbars : bool, optional
            If ``True`` (default), add colourbars to the figure.
        alpha : float, optional
            Patch transparency (0–1).  Defaults to 1.
        coast_on_top : bool, optional
            If ``True``, redraw the coastline on top of the slip patches.

        Returns
        -------
        list
            Matplotlib ``PolyCollection`` objects for each fault plotted.
        """
        assert self.patches is not None, "Need to populate object with patches!"

        if all([bounds is None, self.bounds is not None]):
            bounds = self.bounds

        if all([min_slip_percentile is not None, min_slip_value is None]):
            min_slip = np.percentile(self.patch_slip, min_slip_percentile)
        else:
            min_slip = min_slip_value

        if subplots is not None:
            if isinstance(subplots, str):
                # Assume pickled figure
                with open(subplots, "rb") as pfile:
                    loaded_subplots = pickle.load(pfile)
                fig, background_ax = loaded_subplots
                ax = background_ax["main_figure"]

            else:
                # Assume matplotlib objects
                fig, ax = subplots
                assert isinstance(fig, plt.Figure), "subplots must be a matplotlib figure or a pickled figure"
                assert isinstance(ax, plt.Axes), "subplots must be a matplotlib figure or a pickled figure"

        elif create_background:
            fig, background_ax = plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                                 bounds=bounds, plot_rivers=plot_rivers, plot_lakes=plot_lakes,
                                                 plot_highways=plot_highways, plot_boundaries=plot_boundaries,
                                                 hillshade_cmap=hillshade_cmap, wgs=wgs, land_color=land_color,
                                                 plot_edge_label=plot_edge_label)
            ax = background_ax["main_figure"]
        elif coast_only:

            fig, background_ax = plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                                 bounds=bounds, plot_rivers=False, plot_lakes=False, plot_highways=False,
                                                 plot_boundaries=False, hillshade_cmap=hillshade_cmap, wgs=wgs,
                                                 land_color=land_color, plot_edge_label=plot_edge_label)
            ax = background_ax["main_figure"]


        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(figsize)

        # Find maximum slip for subduction interface

        # Find maximum slip to scale colourbar
        max_slip = 0
        if extra_sub_list is not None:
            sub_list = bruce_subduction + extra_sub_list
        else:
            sub_list = bruce_subduction
        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name in sub_list:
                if plot_zeros:
                    colours = np.zeros(fault.patch_numbers.shape)
                else:
                    colours = np.nan * np.ones(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.patch_numbers:
                        slip_index = np.argwhere(self.patch_numbers == patch_id)[0]
                        if min_slip is not None:
                            if self.patch_slip[slip_index] >= min_slip:
                                colours[local_id] = self.patch_slip[slip_index]
                        else:
                            if self.patch_slip[slip_index] > 0.:
                                colours[local_id] = self.patch_slip[slip_index]

                colour_dic[f_i] = colours
                if np.nanmax(colours) > max_slip:
                    max_slip = np.nanmax(colours)
        max_slip = global_max_sub_slip if global_max_sub_slip > 0 else max_slip

        plots = []

        # Plot subduction interface
        subduction_list = []
        subduction_plot = None
        for f_i, fault in enumerate(self.faults):
            if fault.name in sub_list:
                subduction_list.append(fault.name)
                if plot_log_scale:
                    subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max),
                                                   alpha=alpha)
                else:
                    subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=subduction_cmap, vmin=0., vmax=max_slip, alpha=alpha)
                plots.append(subduction_plot)

        max_slip = 0
        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name not in sub_list:
                if plot_zeros:
                    colours = np.zeros(fault.patch_numbers.shape)
                else:
                    colours = np.nan * np.ones(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.patch_numbers:
                        slip_index = np.argwhere(self.patch_numbers == patch_id)[0]
                        if min_slip is not None:
                            if self.patch_slip[slip_index] >= min_slip:
                                colours[local_id] = self.patch_slip[slip_index]
                        else:
                            if self.patch_slip[slip_index] > 0.:
                                colours[local_id] = self.patch_slip[slip_index]
                colour_dic[f_i] = colours
                if np.nanmax(colours) > max_slip:
                    max_slip = np.nanmax(colours)
        max_slip = global_max_slip if global_max_slip > 0 else max_slip

        crustal_plot = None
        # check for 90 degree dipping faults
        vert_faults = [fault for fault in self.faults if fault.mean_dip == 90.]
        if len(vert_faults) == 0:
            for f_i, fault in enumerate(self.faults):
                if isinstance(fault.trace, LineString):
                    ax.plot(*fault.trace.coords.xy, color='black', linestyle='dashed', linewidth=0.1)
                else:
                    try:
                        merged_coords = [list(geom.coords) for geom in fault.trace.geoms]
                        merged_trace = LineString([trace for sublist in merged_coords for trace in sublist])
                        ax.plot(*merged_trace.coords.xy, color='black', linestyle='dashed', linewidth=0.1)
                    except:
                        pass
                if fault.name not in sub_list:
                    if plot_log_scale:
                        crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                    facecolors=colour_dic[f_i],
                                                    cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max),
                                                    alpha=alpha)
                    else:
                        crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                    facecolors=colour_dic[f_i],
                                                    cmap=crustal_cmap, vmin=0., vmax=max_slip, alpha=alpha)
                    plots.append(crustal_plot)

        elif len(self.faults) == 1:
            fault = self.faults[0]
            f_i = 0
            if isinstance(fault.trace, LineString):
                ax.plot(*fault.trace.coords.xy, color='black', linestyle='dashed', linewidth=0.1)
            else:
                try:
                    merged_coords = [list(geom.coords) for geom in fault.trace.geoms]
                    merged_trace = LineString([trace for sublist in merged_coords for trace in sublist])
                    ax.plot(*merged_trace.coords.xy, color='black', linestyle='dashed', linewidth=0.1)
                except:
                    pass
            if fault.name not in sub_list:
                if plot_log_scale:
                    crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 2], fault.triangles,
                                                facecolors=colour_dic[f_i],
                                                cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max),
                                                alpha=alpha)
                else:
                    crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 2], fault.triangles,
                                                facecolors=colour_dic[f_i],
                                                cmap=crustal_cmap, vmin=0., vmax=max_slip, alpha=alpha)
                plots.append(crustal_plot)
        else:
            for f_i, fault in enumerate(self.faults):
                if isinstance(fault.trace, LineString):
                    ax.plot(*fault.trace.coords.xy, color='black', linestyle='dashed', linewidth=0.1)
                else:
                    try:
                        merged_coords = [list(geom.coords) for geom in fault.trace.geoms]
                        merged_trace = LineString([trace for sublist in merged_coords for trace in sublist])
                        ax.plot(*merged_trace.coords.xy, color='black', linestyle='dashed', linewidth=0.1)
                    except:
                        pass
                if fault.name not in sub_list:
                    if fault in vert_faults:
                        pass
                    else:

                        if plot_log_scale:
                            crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                        facecolors=colour_dic[f_i],
                                                        cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max))
                        else:
                            crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                        facecolors=colour_dic[f_i],
                                                        cmap=crustal_cmap, vmin=0., vmax=max_slip)
                        plots.append(crustal_plot)

        if plot_cbars:
            if any([subplots is None, isinstance(subplots, str)]):
                if plot_log_scale:
                    if subduction_list:
                        sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                        sub_cbar.set_label("Slip (m)")
                    elif crustal_plot is not None:
                        crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                        crust_cbar.set_label("Crustal slip (m)")
                else:
                    if subduction_list:
                        sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                        sub_cbar.set_label("Subduction slip (m)")
                    if crustal_plot is not None:
                        crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                        crust_cbar.set_label("Crustal slip (m)")
        if coast_on_top:
            plot_coast(ax=ax, wgs=wgs, linewidth=0.5, edgecolor='k')

        if title:
            plt.suptitle(title)
        if write is not None:
            fig.savefig(write, dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)

        if show and subplots is None:
            plt.show()

        return plots

    def plot_slip_evolution(self, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", show: bool = True,
                            step_size: int = 1, write: str = None, fps: int = 20, file_format: str = "gif",
                            figsize: tuple = (6.4, 4.8), extra_sub_list: list = None):
        """
        Animate the temporal evolution of slip propagation for this event.

        Produces a ``FuncAnimation`` that steps through rupture time,
        progressively revealing each patch as it slips.  Can be saved
        as a GIF or video file.

        Parameters
        ----------
        subduction_cmap : str, optional
            Matplotlib colourmap for subduction-interface patches.
            Defaults to ``"plasma"``.
        crustal_cmap : str, optional
            Matplotlib colourmap for crustal patches.  Defaults to
            ``"viridis"``.
        show : bool, optional
            If ``True`` (default), display the animation interactively.
        step_size : int, optional
            Time step (s) between animation frames.  Defaults to 1.
        write : str or None, optional
            Output file path prefix (without extension).  If ``None``
            the animation is not saved.
        fps : int, optional
            Frames per second for the saved animation.  Defaults to 20.
        file_format : str, optional
            Output format: ``"gif"``, ``"mov"``, ``"avi"``, or ``"mp4"``.
            Defaults to ``"gif"``.
        figsize : tuple of float, optional
            Figure size in inches ``(width, height)``.
        extra_sub_list : list of str, optional
            Additional fault names to treat as subduction interface.
        """
        assert file_format in ("gif", "mov", "avi", "mp4")
        assert len(self.faults) > 0, "Can't plot an event with no faults."
        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        ax.set_facecolor('w')
        plt.figure(facecolor='w')
        plot_coast(ax, clip_boundary=self.bounds)
        ax.set_aspect("equal")
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

        colour_dic = {}
        timestamps = defaultdict(set)
        subduction_max_slip = 0
        crustal_max_slip = 0

        if extra_sub_list is not None:
            sub_list = [*bruce_subduction, *extra_sub_list]
        else:
            sub_list = bruce_subduction

        subduction_list = []
        for f_i, fault in enumerate(self.faults):
            colours = np.zeros(fault.patch_numbers.shape)
            times = np.zeros(fault.patch_numbers.shape)

            for local_id, patch_id in enumerate(fault.patch_numbers):
                if patch_id in self.patch_numbers:
                    slip_index = np.searchsorted(self.patch_numbers, patch_id)
                    times[local_id] = step_size * np.rint((self.patch_time[slip_index] - self.t0) / step_size)
                    colours[local_id] = self.patch_slip[slip_index]
                    timestamps[times[local_id]].add(f_i)

            colour_dic[f_i] = (colours, times)
            if fault.name in sub_list:
                subduction_list.append(fault.name)
                if max(colours) > subduction_max_slip:
                    subduction_max_slip = max(colours)
            else:
                if max(colours) > crustal_max_slip:
                    crustal_max_slip = max(colours)

        plots = {}
        subduction_plot = None
        for f_i, fault in enumerate(self.faults):
            init_colours = np.zeros(fault.patch_numbers.shape)
            if isinstance(fault.trace, LineString):
                ax.plot(*fault.trace.coords.xy, color='gray', linestyle='dashed')
            else:
                try:
                    merged_coords = [list(geom.coords) for geom in fault.trace.geoms]
                    merged_trace = LineString([trace for sublist in merged_coords for trace in sublist])
                    ax.plot(*merged_trace.coords.xy, color='gray', linestyle='dashed')
                except:
                    pass
            if fault.name in sub_list:
                subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                               facecolors=init_colours,
                                               cmap=subduction_cmap, vmin=0, vmax=subduction_max_slip)
                plots[f_i] = (subduction_plot, init_colours)

        crustal_plot = None
        for f_i, fault in enumerate(self.faults):
            init_colours = np.zeros(fault.patch_numbers.shape)
            if fault.name not in sub_list:
                crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                            facecolors=init_colours,
                                            cmap=crustal_cmap, vmin=0, vmax=crustal_max_slip)
                plots[f_i] = (crustal_plot, init_colours)

        ax_divider = make_axes_locatable(ax)
        ax_time = ax_divider.append_axes("bottom", size="3%", pad=0.5)
        time_slider = Slider(ax_time, 'Time (s)', 0, step_size * round(self.dt / step_size) + step_size,
                             valinit=0, valstep=step_size)

        # Build colorbars
        padding = 0.25
        if subduction_list:
            sub_ax = ax_divider.append_axes("right", size="5%", pad=padding)
            sub_cbar = fig.colorbar(
                subduction_plot, cax=sub_ax)
            sub_cbar.set_label("Subduction slip (m)")
            padding += 0.25

        if crustal_plot is not None:
            crust_ax = ax_divider.append_axes("right", size="5%", pad=padding)
            crust_cbar = fig.colorbar(
                crustal_plot, cax=crust_ax)
            crust_cbar.set_label("Slip (m)")

        def update_plot(num):
            time = time_slider.valmin + num * step_size
            time_slider.set_val(time)

            if time in timestamps:
                for f_i in timestamps[time]:
                    plot, curr_colours = plots[f_i]
                    fault_times = colour_dic[f_i][1]
                    filter_time_indices = np.argwhere(fault_times == time).flatten()
                    curr_colours[filter_time_indices] = colour_dic[f_i][0][filter_time_indices]
                    plot.update({'array': curr_colours})
            elif time == time_slider.valmax:
                for f_i, fault in enumerate(self.faults):
                    plot, curr_colours = plots[f_i]
                    init_colors = np.zeros(fault.patch_numbers.shape)
                    curr_colours[:] = init_colors[:]
                    plot.update({'array': curr_colours})

            fig.canvas.draw_idle()

        frames = int((time_slider.valmax - time_slider.valmin) / step_size) + 1
        animation = FuncAnimation(fig, update_plot, interval=50, frames=frames)

        if write is not None:
            writer = PillowWriter(fps=fps) if file_format == "gif" else FFMpegWriter(fps=fps)
            animation.save(f"{write}.{file_format}", writer, savefig_kwargs=dict(facecolor='w'))

        if show:
            plt.show()

    def find_surface_faults(self,fault_model: RsqSimMultiFault,min_slip: float =0.1, method: str = 'vertex',
                                      n_patches: int = 1, max_depth: float = -1000., faults2ignore: [list,str] ='hikurangi'):
        """
        Identify fault segments with surface-rupturing patches.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing the patch dictionary.
        min_slip : float, optional
            Minimum slip (m) for a patch to count as ruptured.
            Defaults to 0.1 m.
        method : str, optional
            Depth criterion: ``"vertex"`` uses the shallowest vertex
            of a patch; ``"centroid"`` uses the patch centroid depth.
            Defaults to ``"vertex"``.
        n_patches : int, optional
            Minimum number of qualifying surface patches required for
            a fault to be included in the output.  Defaults to 1.
        max_depth : float, optional
            Depth threshold (m, negative downward).  Patches shallower
            than this are considered surface-rupturing.  Defaults to
            -1000 m.
        faults2ignore : list of str or str, optional
            Fault name(s) to exclude from the search (e.g. subduction
            interface).  Defaults to ``"hikurangi"``.

        Returns
        -------
        list of str
            Names of fault segments that have surface rupture.
        """

        assert method in ['centroid', 'vertex'], "Method must be centroid or vertex"
        assert max_depth < 0., "depths should be negative"
        if issubclass(type(faults2ignore),str):
            faults2ignore =[faults2ignore]

        surface_faults = []
        for fault in self.faults:
            if not fault.name in faults2ignore:
                surface_patches = []
                for patch_id in fault.patch_numbers:
                    if patch_id in self.patch_numbers:
                        patch = fault.patch_dic[patch_id]

                        if method == 'vertex':
                            patch_zs = patch.vertices.flatten()[[2, 5, 8]]
                            patch_z = np.max(patch_zs)  # use max because depths are negative
                        elif method == 'centroid':
                            patch_z = patch.centre[2]
                        else:
                            AssertionError('method must be vertex or centroid')

                        if patch_z > max_depth:
                            patch_ev_index = np.searchsorted(self.patch_numbers, patch_id)
                            patch_slip = self.patch_slip[patch_ev_index]
                            if patch_slip >= min_slip:
                                surface_patches.append(patch_ev_index)

                if len(surface_patches) >= n_patches:
                    surface_faults.append(fault.name)


        return surface_faults

    def split_by_fault(self, fault_model: RsqSimMultiFault, min_slip: float = 0.1, min_patches: int = 1):
        """
        Create per-fault sub-event objects for this event.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model used to resolve patch IDs and fault names.
        min_slip : float, optional
            Minimum moment (N·m) for a fault to be included (passed as
            ``min_m0`` to :meth:`make_fault_moment_dict`).  Defaults to
            0.1.
        min_patches : int, optional
            Unused; retained for API compatibility.

        Returns
        -------
        dict
            Mapping of fault name (str) to its corresponding
            :class:`RsqSimEvent` sub-event.
        """
        assert self.faults is not None, "Event has no faults, can't split by fault"
        subevents = {}
        moment_dict = self.make_fault_moment_dict(fault_model=fault_model, min_m0=min_slip)
        for i, fault in enumerate(self.faults):
            if fault.name in moment_dict.keys():
                sub_m0 = moment_dict[fault.name]
                sub_mw = m0_to_mw(sub_m0)
                fault_patches = np.array(list(fault.patch_dic.keys()))
                fault_patch_numbers = self.patch_numbers[np.isin(self.patch_numbers, fault_patches)]
                fault_patch_slip = self.patch_slip[np.isin(self.patch_numbers, fault_patches)]
                fault_patch_time = self.patch_time[np.isin(self.patch_numbers, fault_patches)]
                sub_event = self.from_earthquake_list(t0=self.t0, m0=sub_m0, mw=sub_mw, x=self.x, y=self.y, z=self.z,
                                                      area=self.area, dt=self.dt,
                                                      patch_numbers=fault_patch_numbers, patch_slip=fault_patch_slip,
                                                      patch_time=fault_patch_time, fault_model=fault_model, event_id=i)
                subevents[fault.name] = sub_event

        return subevents


    def slip_dist_array(self, include_zeros: bool = True, min_slip_percentile: float = None,
                        min_slip_value: float = None, nztm_to_lonlat: bool = False):
        """
        Build a dense array of the slip distribution across all patches.

        Each row corresponds to one triangular patch and contains the
        three vertex coordinates followed by slip, rake, and rupture time.

        Parameters
        ----------
        include_zeros : bool, optional
            If ``True`` (default), include zero-slip patches (patches
            on involved faults that did not slip in this event).
        min_slip_percentile : float or None, optional
            If provided (and ``min_slip_value`` is ``None``), patches
            with slip below this percentile are excluded or zeroed.
        min_slip_value : float or None, optional
            If provided, patches with slip below this value (m) are
            excluded or zeroed.
        nztm_to_lonlat : bool, optional
            If ``True``, transform vertex coordinates from NZTM to
            WGS84 longitude/latitude before including them.

        Returns
        -------
        numpy.ndarray of shape (n_patches, 12)
            Columns: ``[x1,y1,z1, x2,y2,z2, x3,y3,z3, slip_m, rake_deg, time_s]``.
        """
        all_patches = []
        if all([min_slip_percentile is not None, min_slip_value is None]):
            min_slip = np.percentile(self.patch_slip, min_slip_percentile)
        else:
            min_slip = min_slip_value

        for fault in self.faults:
            for patch_id in fault.patch_numbers:
                if patch_id in self.patch_numbers:
                    patch = fault.patch_dic[patch_id]
                    if nztm_to_lonlat:
                        triangle_corners = patch.vertices_lonlat.flatten()
                    else:
                        triangle_corners = patch.vertices.flatten()
                    slip_index = np.searchsorted(self.patch_numbers, patch_id)
                    time = self.patch_time[slip_index] - self.t0
                    slip_mag = self.patch_slip[slip_index]
                    if min_slip is not None:
                        if slip_mag >= min_slip:
                            patch_line = np.hstack([triangle_corners, np.array([slip_mag, patch.rake, time])])
                            all_patches.append(patch_line)
                        elif include_zeros:
                            patch = fault.patch_dic[patch_id]
                            patch_line = np.hstack([triangle_corners, np.array([0., 0., 0.])])
                            all_patches.append(patch_line)
                    else:
                        patch_line = np.hstack([triangle_corners, np.array([slip_mag, patch.rake, time])])
                        all_patches.append(patch_line)
                elif include_zeros:
                    patch = fault.patch_dic[patch_id]
                    if nztm_to_lonlat:
                        triangle_corners = patch.vertices_lonlat.flatten()
                    else:
                        triangle_corners = patch.vertices.flatten()
                    patch_line = np.hstack([triangle_corners, np.array([0., 0., 0.])])
                    all_patches.append(patch_line)
        return np.array(all_patches)

    def slip_dist_bounds(self, include_zeros: bool = True, min_slip_percentile: float = None,
                            min_slip_value: float = None, nztm_to_lonlat: bool = False):
        """
        Compute the bounding box of the slip distribution array.

        Parameters
        ----------
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_array`.
        nztm_to_lonlat : bool, optional
            Passed to :meth:`slip_dist_array`.

        Returns
        -------
        tuple of float
            ``(x_min, y_min, x_max, y_max)`` extents of the slip
            distribution patch vertices.
        """
        slip_dist_array = self.slip_dist_array(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                               min_slip_value=min_slip_value, nztm_to_lonlat=nztm_to_lonlat)
        min_x = np.min(slip_dist_array[:, [0, 3, 6]])
        max_x = np.max(slip_dist_array[:, [0, 3, 6]])
        min_y = np.min(slip_dist_array[:, [1, 4, 7]])
        max_y = np.max(slip_dist_array[:, [1, 4, 7]])
        return min_x, min_y, max_x, max_y

    def slip_dist_to_mesh(self, include_zeros: bool = True, min_slip_percentile: float = None,
                          min_slip_value: float = None, nztm_to_lonlat: bool = False):
        """
        Convert the slip distribution to a meshio Mesh with cell data.

        Parameters
        ----------
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_array`.
        nztm_to_lonlat : bool, optional
            Passed to :meth:`slip_dist_array`.

        Returns
        -------
        meshio.Mesh
            Triangulated mesh with ``cell_data`` containing ``"slip"``,
            ``"rake"``, and ``"time"`` arrays.
        """
        slip_dist_array = self.slip_dist_array(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                               min_slip_value=min_slip_value)
        mesh = array_to_mesh(slip_dist_array[:, :9])
        data_dic = {}
        for label, index in zip(["slip", "rake", "time"], [9, 10, 11]):
            data_dic[label] = slip_dist_array[:, index]

        mesh.cell_data = data_dic

        return mesh

    def slip_dist_to_vtk(self, vtk_file: str, include_zeros: bool = True, min_slip_percentile: float = None,
                         min_slip_value: float = None):
        """
        Write the slip distribution to a VTK file.

        Parameters
        ----------
        vtk_file : str
            Output VTK file path.
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_to_mesh`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_to_mesh`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_to_mesh`.
        """
        mesh = self.slip_dist_to_mesh(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                      min_slip_value=min_slip_value)
        mesh.write(vtk_file, file_format="vtk")

    def slip_dist_to_obj(self, obj_file: str, include_zeros: bool = True, min_slip_percentile: float = None,
                         min_slip_value: float = None):
        """
        Write the slip distribution to a Wavefront OBJ file.

        Parameters
        ----------
        obj_file : str
            Output OBJ file path.
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_to_mesh`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_to_mesh`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_to_mesh`.
        """
        mesh = self.slip_dist_to_mesh(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                      min_slip_value=min_slip_value)
        mesh.write(obj_file, file_format="obj")

    def slip_dist_to_txt(self, txt_file, include_zeros: bool = True, min_slip_percentile: float = None,
                         min_slip_value: float = None, nztm_to_lonlat: bool = False):
        """
        Write the slip distribution to a space-delimited text file.

        Each row contains the three vertex coordinates of a patch
        followed by slip, rake, and rupture time.

        Parameters
        ----------
        txt_file : str or path-like
            Output text file path.
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_array`.
        nztm_to_lonlat : bool, optional
            If ``True``, vertex coordinates are in WGS84 lon/lat and
            the header is adjusted accordingly.
        """
        if nztm_to_lonlat:
            header = "lon1 lat1 z1 lon2 lat2 z2 lon3 lat3 z3 slip_m rake_deg time_s"
        else:
            header = "x1 y1 z1 x2 y2 z2 x3 y3 z3 slip_m rake_deg time_s"
        slip_dist_array = self.slip_dist_array(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                               min_slip_value=min_slip_value, nztm_to_lonlat=nztm_to_lonlat)
        np.savetxt(txt_file, slip_dist_array, fmt="%.6f", delimiter=" ", header=header)

    def slip_dist_to_gdf(self, gdf_file: str, include_zeros: bool = True, min_slip_percentile: float = None,
                         min_slip_value: float = None, nztm_to_lonlat: bool = False, crs="2193"):
        """
        Build a GeoDataFrame of the slip distribution.

        Creates a GeoDataFrame where each row is a triangular patch
        polygon with slip, rake, and time attributes.

        Parameters
        ----------
        gdf_file : str or None
            Unused; retained for API compatibility.
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_array`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_array`.
        nztm_to_lonlat : bool, optional
            If ``True``, reproject the result from NZTM to WGS84.
        crs : str, optional
            CRS for the output GeoDataFrame.  Defaults to ``"2193"``
            (NZTM / EPSG:2193).

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with columns ``["slip", "rake", "time"]`` and
            a ``geometry`` column of triangular patch polygons.
        """
        slip_dist_array = self.slip_dist_array(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                               min_slip_value=min_slip_value, nztm_to_lonlat=nztm_to_lonlat)
        geometry = [Polygon([(slip_dist_array[i, 0], slip_dist_array[i, 1]),
                             (slip_dist_array[i, 3], slip_dist_array[i, 4]),
                             (slip_dist_array[i, 6], slip_dist_array[i, 7])]) for i in range(len(slip_dist_array))]
        gdf = gpd.GeoDataFrame(slip_dist_array[:, 9:], columns=["slip", "rake", "time"], geometry=geometry, crs=crs)
        if nztm_to_lonlat:
            assert crs == "2193"
            gdf.to_crs("EPSG:4326", inplace=True)

        return gdf

    def slip_dist_to_geojson(self, geojson_file: str, include_zeros: bool = True, min_slip_percentile: float = None,
                             min_slip_value: float = None, nztm_to_lonlat: bool = False):
        """
        Write the slip distribution to a GeoJSON file.

        Parameters
        ----------
        geojson_file : str
            Output GeoJSON file path.
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_to_gdf`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_to_gdf`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_to_gdf`.
        nztm_to_lonlat : bool, optional
            If ``True``, reproject patches to WGS84 before writing.
        """
        gdf = self.slip_dist_to_gdf(gdf_file=None, include_zeros=include_zeros,
                                    min_slip_percentile=min_slip_percentile, min_slip_value=min_slip_value,
                                    nztm_to_lonlat=nztm_to_lonlat)
        gdf.to_file(geojson_file, driver="GeoJSON")

    def slip_dist_to_shapefile(self, shapefile_file: str, include_zeros: bool = True, min_slip_percentile: float = None,
                               min_slip_value: float = None, nztm_to_lonlat: bool = False):
        """
        Write the slip distribution to an ESRI Shapefile.

        Parameters
        ----------
        shapefile_file : str
            Output shapefile path (without extension).
        include_zeros : bool, optional
            Passed to :meth:`slip_dist_to_gdf`.
        min_slip_percentile : float or None, optional
            Passed to :meth:`slip_dist_to_gdf`.
        min_slip_value : float or None, optional
            Passed to :meth:`slip_dist_to_gdf`.
        nztm_to_lonlat : bool, optional
            If ``True``, reproject patches to WGS84 before writing.
        """
        gdf = self.slip_dist_to_gdf(gdf_file=None, include_zeros=include_zeros,
                                    min_slip_percentile=min_slip_percentile, min_slip_value=min_slip_value,
                                    nztm_to_lonlat=nztm_to_lonlat)
        gdf.to_file(shapefile_file)

    def slip_dist_to_gmt(self, fault_model: RsqSimMultiFault, gmt_prefix: str,
                         min_slip_value: float = None, nztm_to_lonlat: bool = False, subduction_names: Iterable = ("hikkerm", "puysegur")):
        """
        Write the slip distribution to GMT multi-segment text files.

        Writes two files: ``<gmt_prefix>_crustal.gmt`` for crustal faults
        and ``<gmt_prefix>_subduction.gmt`` for subduction-interface faults.
        Each patch is written as a ``>-Z<slip>`` segment header followed
        by vertex coordinates.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing the patch dictionary.
        gmt_prefix : str
            Prefix for the output GMT file names.
        min_slip_value : float or None, optional
            If set, only patches with slip >= this value (m) are written.
        nztm_to_lonlat : bool, optional
            Unused; reserved for future coordinate transformation.
        subduction_names : iterable of str, optional
            Fault names to classify as subduction interface.  Defaults
            to ``("hikkerm", "puysegur")``.
        """
        crustal_faults = [fault for fault in self.faults if fault.name not in subduction_names]
        subduction_faults = [fault for fault in self.faults if fault.name in subduction_names]

        if crustal_faults:
            crustal_patch_ids = []
            crustal_patch_slip = []
            crustal_patch_depth = []
            
            for fault in crustal_faults:
                fault_patches = fault.patch_numbers
                event_fault_indices = np.isin(self.patch_numbers, fault_patches)
                event_fault_slip = self.patch_slip[event_fault_indices]
                if min_slip_value is not None:
                    event_gt_min_slip_indices = event_fault_slip > min_slip_value
                    filtered_patch_indices = np.where(event_gt_min_slip_indices & event_fault_indices)[0]
                else:
                    filtered_patch_indices = np.where(event_fault_indices)[0]
                crustal_patch_ids.extend(self.patch_numbers[filtered_patch_indices])
                crustal_patch_slip.extend(self.patch_slip[filtered_patch_indices])
                for slip_index in filtered_patch_indices:
                    patch_id = self.patch_numbers[slip_index]
                    patch = fault.patch_dic[patch_id]
                    depth = np.mean(patch.vertices[:, 2])
                    crustal_patch_depth.append(depth)
                
            if len(crustal_patch_ids) > 0:
                sorted_depths = np.argsort(crustal_patch_depth)
                crustal_patch_ids = np.array(crustal_patch_ids)[sorted_depths]
                crustal_patch_slip = np.array(crustal_patch_slip)[sorted_depths]
                crustal_patch_depth = np.array(crustal_patch_depth)[sorted_depths]

                with open(f"{gmt_prefix}_crustal.gmt", "w") as f:
                    for patch_id, slip, depth in zip(crustal_patch_ids, crustal_patch_slip, crustal_patch_depth):
                        patch = fault_model.patch_dic[patch_id]
                        f.write(f">-Z{slip:.6f}\n")
                        for vertex in patch.vertices:
                            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        if subduction_faults:
            with open(f"{gmt_prefix}_subduction.gmt", "w") as f:
                for fault in subduction_faults:
                    fault_patches = fault.patch_numbers
                    event_fault_indices = np.isin(self.patch_numbers, fault_patches)
                    event_fault_slip = self.patch_slip[event_fault_indices]
                    if min_slip_value is not None:
                        event_gt_min_slip_indices = event_fault_slip > min_slip_value
                        filtered_patch_indices = np.where(event_gt_min_slip_indices & event_fault_indices)[0]
                    else:
                        filtered_patch_indices = np.where(event_fault_indices)[0]
                    for slip_index in filtered_patch_indices:
                        patch_id = self.patch_numbers[slip_index]
                        patch = fault.patch_dic[patch_id]
                        slip = self.patch_slip[slip_index]

                        f.write(f">-Z{slip:.6f}\n")
                        for vertex in patch.vertices:
                            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

    def discretize_tiles(self, tile_list: list[Polygon], probability: float, rake: float):
        """
        Select tiles that overlap the rupture exterior by at least 50 %.

        Parameters
        ----------
        tile_list : list of Polygon
            Candidate rectangular tiles in NZTM coordinates.
        probability : float
            Unused; retained for API compatibility with
            :meth:`discretize_openquake`.
        rake : float
            Unused; retained for API compatibility.

        Returns
        -------
        geopandas.GeoSeries
            GeoSeries (EPSG:2193) of tiles where the intersection with
            the rupture exterior covers at least half the tile area.
        """
        included_tiles = []

        for tile in tile_list:
            overlapping = tile.intersects(self.exterior)
            if overlapping:
                intersection = tile.intersection(self.exterior)
                if intersection.area >= 0.5 * tile.area:
                    included_tiles.append(tile)

        out_gs = gpd.GeoSeries(included_tiles, crs=2193)
        return out_gs

    def discretize_openquake(self, tile_list: list[Polygon], probability: float, rake: float):
        """
        Build an OpenQuake multi-planar rupture from overlapping tiles.

        Selects tiles from ``tile_list`` that overlap the rupture exterior
        by at least 50 %, then packages them as an
        :class:`OpenQuakeMultiSquareRupture`.

        Parameters
        ----------
        tile_list : list of Polygon
            Candidate rectangular tiles in NZTM coordinates.
        probability : float
            Probability of occurrence passed to
            :class:`OpenQuakeMultiSquareRupture`.
        rake : float
            Mean rake (degrees) passed to
            :class:`OpenQuakeMultiSquareRupture`.

        Returns
        -------
        OpenQuakeMultiSquareRupture or None
            Rupture object if any tiles overlap; ``None`` otherwise.
        """
        included_tiles = []

        for tile in tile_list:
            overlapping = tile.intersects(self.exterior)
            if overlapping:
                intersection = tile.intersection(self.exterior)
                if intersection.area >= 0.5 * tile.area:
                    included_tiles.append(tile)

        out_gs = gpd.GeoSeries(included_tiles, crs=2193)
        out_gs_wgs = out_gs.to_crs(epsg=4326)
        if out_gs_wgs.size > 0:
            oq_rup = OpenQuakeMultiSquareRupture(list(out_gs.geometry), magnitude=self.mw, rake=rake,
                                                 hypocentre=np.array([self.x, self.y, self.z]),
                                                 event_id=self.event_id, probability=probability)
            return oq_rup

        else:
            return

    def discretize_openquake_ktree(self, fault_model: RsqSimMultiFault, quads_dict: dict, probability: float,
                                   subduction_names: Iterable = ("hikkerm", "puysegur"), min_moment = 1.e18,
                                   min_slip = 0.1, tile_size: float = 5000., write_mesh: bool = False,
                                   write_geojson: bool = False, xml_dir: str = None, threshold: float = 0.5):
        """
        Build OpenQuake XML ruptures using a KD-tree tile assignment.

        Uses :meth:`slip_dist_quads_ktree` to find ruptured quadrilateral
        tiles per fault, then writes separate XML files for crustal and
        subduction components.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch and fault dictionaries.
        quads_dict : dict
            Mapping of fault segment name to a ``(n, 4, 3)`` array of
            pre-computed quadrilateral tile corner coordinates.
        probability : float
            Probability of occurrence for the OpenQuake rupture.
        subduction_names : iterable of str, optional
            Fault names classified as subduction interface.
        min_moment : float, optional
            Minimum moment (N·m) for a fault to be included.
        min_slip : float, optional
            Minimum slip (m) for a patch to count as ruptured.
        tile_size : float, optional
            Tile size hint (m); unused here but passed through for
            context.
        write_mesh : bool, optional
            If ``True``, also write VTK mesh files for each component.
        write_geojson : bool, optional
            If ``True``, also write GeoJSON files for each component.
        xml_dir : str or None, optional
            Directory in which to write output files.  Defaults to the
            current working directory.
        threshold : float, optional
            Fraction of patches in a tile that must be ruptured for the
            tile to be included.  Passed to :meth:`slip_dist_quads_ktree`.
        """
        tiles_dict = self.slip_dist_quads_ktree(quads_dict=quads_dict, fault_model=fault_model,min_moment=min_moment,
                                                min_slip=min_slip, threshold_for_inclusion=threshold)
        crustal_faults = [key for key in tiles_dict.keys() if key not in subduction_names]
        if crustal_faults:
            if all([tiles_dict[key].size == 0 for key in crustal_faults]):
                crustal_faults = []
        subduction_faults = [key for key in tiles_dict.keys() if key in subduction_names]
        if subduction_faults:
            if all([tiles_dict[key].size == 0 for key in subduction_faults]):
                subduction_faults = []

        if subduction_faults:
            subduction_tiles = np.vstack([tiles_dict[key] for key in subduction_faults])
            if write_mesh:
                mesh = quads_to_vtk(subduction_tiles)
                vtk_name = f"event_{self.event_id}_subduction.vtk"
                if xml_dir is not None:
                    vtk_name = os.path.join(xml_dir, vtk_name)
                mesh.write(vtk_name, file_format="vtk")

            subduction_tiles_gs = gpd.GeoSeries([Polygon(subduction_tile) for subduction_tile in subduction_tiles], crs=2193)
            if write_geojson:
                geojson_name = f"event_{self.event_id}_subduction.geojson"
                if xml_dir is not None:
                    geojson_name = os.path.join(xml_dir, geojson_name)
                subduction_tiles_gs.to_file(geojson_name, driver="GeoJSON")
            subduction_component = self.get_subduction_component(fault_model=fault_model, subduction_names=subduction_names,
                                                                 min_moment=min_moment, min_slip=min_slip)
            if subduction_component is not None:
                subduction_mw, subduction_rake, subduction_centroid = subduction_component
                oq_rup = OpenQuakeMultiSquareRupture(list(subduction_tiles_gs.geometry), magnitude=subduction_mw,
                                                        rake=subduction_rake, hypocentre=subduction_centroid,
                                                        event_id=self.event_id, probability=probability)
                out_name = f"event_{self.event_id}_subduction.xml"
                if xml_dir is not None:
                    out_file = os.path.join(xml_dir, out_name)
                else:
                    out_file = out_name

                oq_rup.to_oq_xml(out_file)

        if crustal_faults:
            crustal_tiles = np.vstack([tiles_dict[key] for key in crustal_faults if tiles_dict[key].size > 0])
            if write_mesh:
                mesh = quads_to_vtk(crustal_tiles)
                vtk_name = f"event_{self.event_id}_crustal.vtk"
                if xml_dir is not None:
                    vtk_name = os.path.join(xml_dir, vtk_name)
                mesh.write(vtk_name, file_format="vtk")

            crustal_tiles_gs = gpd.GeoSeries([Polygon(crustal_tile) for crustal_tile in crustal_tiles], crs=2193)
            if write_geojson:
                geojson_name = f"event_{self.event_id}_crustal.geojson"
                if xml_dir is not None:
                    geojson_name = os.path.join(xml_dir, geojson_name)
                crustal_tiles_gs.to_file(geojson_name, driver="GeoJSON")
            crustal_component = self.get_crustal_component(fault_model=fault_model, crustal_names=crustal_faults,
                                                           min_moment=min_moment, min_slip=min_slip)
            if crustal_component is not None and not crustal_tiles_gs.is_empty.all():
                crustal_mw, crustal_rake, crustal_centroid = crustal_component
                oq_rup = OpenQuakeMultiSquareRupture(list(crustal_tiles_gs.geometry), magnitude=crustal_mw,
                                                        rake=crustal_rake, hypocentre=crustal_centroid,
                                                        event_id=self.event_id, probability=probability)
                out_name = f"event_{self.event_id}_crustal.xml"
                if xml_dir is not None:
                    out_file = os.path.join(xml_dir, out_name)
                else:
                    out_file = out_name

                oq_rup.to_oq_xml(out_file)









    def event_to_json(self, fault_model: RsqSimMultiFault, path2cfm: str, catalogue_version: str = 'v1',
                      xml_dir: str = 'OQ-events', wgs84: bool = False, subd_tile_size: float = 15000.,
                      tile_size: float = 5000., tectonic_region: str = "NZ"):
        """
        Export event slip distribution to a ShakeMap-compatible GeoJSON file.

        Discretises the rupturing faults into rectangular tiles, selects
        those near slipping patches, converts coordinates to WGS84, and
        writes a GeoJSON file with metadata to
        ``<xml_dir>/event_<id>/<id>.json``.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model used to merge and discretise fault segments.
        path2cfm : str
            Path to the Community Fault Model directory (required for
            ``catalogue_version="v2"`` name lookups).
        catalogue_version : str, optional
            ``"v1"`` or ``"v2"``.  Affects which fault name dictionary
            is used.  Defaults to ``"v1"``.
        xml_dir : str, optional
            Output directory.  Defaults to ``"OQ-events"``.
        wgs84 : bool, optional
            If ``True``, hypocentre coordinates are already in WGS84
            (lon/lat).  If ``False`` (default), NZTM coordinates are
            transformed to WGS84.
        subd_tile_size : float, optional
            Tile size (m) used when discretising subduction faults.
            Defaults to 15000 m.
        tile_size : float, optional
            Tile size (m) used when discretising crustal faults.
            Defaults to 5000 m.
        tectonic_region : str, optional
            Tectonic region tag written into the metadata.  Defaults to
            ``"NZ"``.
        """
        # setup
        assert os.path.exists(path2cfm), "Path to CFM does not exist"

        if catalogue_version == 'v2':
            fault_model.make_v2_name_dic(path2cfm=path2cfm)
            name_dict = fault_model.v2_name_dic
        else:
            name_dict = dict(zip(fault_model.names, fault_model.names))

        # create output directory if needed
        outdir = os.path.join(xml_dir, f'event_{self.event_id}')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        ### create metadata string for event
        # start with timestamp (not sure this is really necessary)
        ev_time = self.t0
        yrs = f"{np.floor(ev_time / (365.25 * 24. * 3600.)):.0f}"
        months = f"{np.floor((ev_time - (float(yrs) * 365.25 * 24. * 3600.)) / (3600. * 24. * 30.4)):02.0f}"
        remain_ym = ev_time - (float(yrs) * (365.25 * 24. * 3600.) + float(months) * (3600. * 24. * 30.4))
        days = np.floor(remain_ym / (24. * 3600.))
        hours = np.floor((remain_ym - days * (24. * 3600.)) / 3600.)
        mins = np.floor((remain_ym - days * (24. * 3600.) - (hours * 3600.)) / 60.)
        secs = remain_ym - days * (24. * 3600.) - (hours * 3600.) - mins * 60.
        timestring = f"{yrs}-{months}-{days:02.0f}T{hours:02.0f}:{mins:02.0f}:{secs:02.6f}Z"
        # location
        if not wgs84:
            lon, lat = transformer_nztm2wgs.transform(self.x, self.y)
        else:
            lon = self.x
            lat = self.y

        metadata = {"lat": np.round(lat, decimals=3), "id": f"rsq{catalogue_version}{self.event_id}", "mech": "ALL",
                    "mag": self.mw, "rake": self.mean_rake, "reference": f"rsqsim_{catalogue_version}",
                    "netid": f"{tectonic_region}", "depth": f'{self.z / 1000.:.1f}',
                    "network": f"rsqsim_{catalogue_version}", "locstring": f"{tectonic_region}",
                    "time": f"{timestring}", "lon": np.round(lon, decimals=3),
                    "productcode": f"rsq{catalogue_version}{self.event_id}"}

        ### convert faults to quadrilaterals
        # which faults are involved in this event?
        faults = RsqSimMultiFault(self.faults)
        faultNames = faults.names
        # find corresponding larger/cfm faults
        allFaults = np.unique([name_dict[name] for name in faultNames])
        subdFaults = np.unique([name_dict[name] for name in faultNames if
                                fnmatch.fnmatch(name, "*puy*") or fnmatch.fnmatch(name, "*hik*")])

        # create empty list to store polygons
        poly_list = []

        # iterate over faults which participate in the event
        for fName in allFaults:
            try:
                # need to find all parts of the fault, then later select those which have non-zero slip
                fault_merged = fault_model.merge_segments(fName, name_dict=name_dict, fault_name=fName)
                if len(subdFaults) > 0 and fName in subdFaults:
                    # find average dip using subduction tile size
                    dip_angle = fault_merged.get_average_dip(subd_tile_size)
                    # Discretize into rectangular tiles
                    new_fault_rect = fault_merged.discretize_rectangular_tiles(tile_size=subd_tile_size)
                else:
                    # average dip
                    dip_angle = fault_merged.get_average_dip()
                    # Discretize into rectangular tiles
                    new_fault_rect = fault_merged.discretize_rectangular_tiles(tile_size=tile_size)

                # want to discard patches which don't slip in the event
                for quad in new_fault_rect:
                    # find nearest triangular patches (to check slip is non-zero on the quads which get passed to the json)
                    approx_centroid = np.mean(quad, axis=0)
                    # find closest patches is only in 2d so then need to check depth
                    nearest_patches_ids = faults.find_closest_patches(approx_centroid[0], approx_centroid[1])
                    nearest_patches = [faults.patch_dic[patch_id] for patch_id in nearest_patches_ids]
                    min_z_diff = min([np.abs(patch.centre[2] - approx_centroid[2]) for patch in nearest_patches])
                    nearest_patches_z = [patch for patch in nearest_patches if
                                         np.abs(patch.centre[2] - approx_centroid[2]) == min_z_diff]
                    patch_ids = [patch.patch_number for patch in nearest_patches_z]
                    # find associated slip
                    slip = 0.
                    for patch_id in patch_ids:
                        if patch_id in self.patch_numbers:
                            slip_index = np.searchsorted(self.patch_numbers, patch_id)
                            slip_mag = self.patch_slip[slip_index]
                            slip += slip_mag

                    mean_slip = slip / len(patch_ids)

                    # check slip isn't 0/ less than a mm
                    if mean_slip > 1.e-3:
                        # convert coordinates to lat lon if needed
                        if not wgs84:
                            x2, y2 = transformer_nztm2wgs.transform(quad[:, 0], quad[:, 1])
                        else:
                            x2, y2 = quad[:, 0], quad[:, 1]
                        # and round to 3dp
                        quad[:, 0] = np.round(x2, decimals=3)
                        quad[:, 1] = np.round(y2, decimals=3)

                        # need depths to be positive and in km
                        new_depths = np.zeros(np.shape(quad[:, 2]))
                        for i, depth in enumerate(quad[:, 2]):
                            # prevent 0s being written as -0.0
                            if isclose(depth, 0.0, abs_tol=0.2):
                                new_depths[i] = 0.
                            else:
                                new_depths[i] = np.round(-1. * depth / 1000., decimals=2)

                        quad[:, 2] = new_depths
                        # sort into correct order for json (shallowest points first)
                        quad_sorted = np.sort(quad, axis=0)
                        poly_list.append(Polygon(quad_sorted))
            except:
                print(f"{fName} could not be discretised - event slip distribution will be incomplete")
        all_segs = MultiPolygon(poly_list)
        polys = gpd.GeoSeries(all_segs)
        # write to initial json
        polys.to_file(os.path.join(outdir, f'{self.event_id}.json'), driver='GeoJSON')

        # read back in to edit properties
        with open(os.path.join(outdir, f'{self.event_id}.json'), 'r') as jfile:
            pjson = json.load(jfile)
        pjson['metadata'] = metadata
        pjson['features'][0]['properties'] = {"rupture type": "rupture extent"}
        # hack to remove extra set of brackets
        pjson['features'][0]['geometry']['coordinates'] = [
            [item[0] for item in pjson['features'][0]['geometry']['coordinates'][:]]]

        # and write back out to json
        with open(os.path.join(outdir, f'{self.event_id}.json'), 'w') as jfile:
            json.dump(pjson, jfile)

    def event_to_OQ_xml(self, fault_model: RsqSimMultiFault, path2cfm: str, catalogue_version: str = 'v2',
                        xml_dir: str = 'OQ_events',
                        subd_tile_size: float = 15000., tile_size: float = 5000., probability: float = 0.9,
                        tectonic_region: str = 'NZ',min_mag: float=6.0,hypocentre: list = None,nztm2wgs: bool = True):
        """
        Export the event as an OpenQuake multi-planar rupture XML file.

        Discretises faults exceeding ``min_mag`` into rectangular tiles,
        selects tiles near ruptured patches, and writes an OpenQuake NRML
        XML file to ``<xml_dir>/<event_id>/event_<event_id>.xml``.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for segment merging and discretisation.
        path2cfm : str
            Path to the Community Fault Model directory.
        catalogue_version : str, optional
            ``"v1"`` or ``"v2"``.  Defaults to ``"v2"``.
        xml_dir : str, optional
            Output directory.  Defaults to ``"OQ_events"``.
        subd_tile_size : float, optional
            Tile size (m) for subduction faults.  Defaults to 15000 m.
        tile_size : float, optional
            Tile size (m) for crustal faults.  Defaults to 5000 m.
        probability : float, optional
            Probability of occurrence for the rupture.  Defaults to 0.9.
        tectonic_region : str, optional
            Tectonic region tag.  Defaults to ``"NZ"``.
        min_mag : float, optional
            Only include faults whose moment corresponds to at least
            this magnitude.  Defaults to 6.0.
        hypocentre : list or None, optional
            ``[x, y, z]`` hypocentre override.  If ``None``, uses the
            event hypocentre.
        nztm2wgs : bool, optional
            If ``True`` (default), convert NZTM coordinates to WGS84
            before writing.
        """
        assert os.path.exists(path2cfm), "Path to CFM does not exist"
        if catalogue_version == 'v2':
            fault_model.make_v2_name_dic(path2cfm=path2cfm)
            name_dict = fault_model.v2_name_dic
        else:
            name_dict = dict(zip(fault_model.names, fault_model.names))
        unique_names = set(name_dict.values())

        # which faults are involved in this event?
        M0s = self.make_fault_moment_dict(fault_model=fault_model, by_cfm_names=False)
        min_M0=10.**(1.5*(min_mag+6.03))
        faults = RsqSimMultiFault(self.faults)
        faultNames = [fault.name for fault in self.faults]
        allFaults = []

        for fault in faultNames:
            print(fault, M0s[fault])
            if M0s[fault] > min_M0:
                allFaults.append(fault)

        subdFaults = np.unique([name_dict[name] for name in faultNames if
                                fnmatch.fnmatch(name, "*puysegar*") or fnmatch.fnmatch(name, "*hikurangi*")])

        outdir = os.path.join(xml_dir, f'{self.event_id}')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        poly_list = []
        for fName in allFaults:
            try:
                # print(fName)
                # need to find all parts of the fault, then later select those which have non-zero slip
                fault_merged = fault_model.merge_segments(fName, name_dict=name_dict, fault_name=fName)

                if len(subdFaults) > 0 and fName in subdFaults:
                    dip_angle = fault_merged.get_average_dip(subd_tile_size)
                    new_fault_rect = fault_merged.discretize_rectangular_tiles(tile_size=subd_tile_size)
                else:
                    # Average dip
                    dip_angle = fault_merged.get_average_dip()
                    # Discretize into rectangular tiles
                    new_fault_rect = fault_merged.discretize_rectangular_tiles(tile_size=tile_size)

                # want to discard patches which don't slip in the event
                for quad in new_fault_rect:
                    # find nearest patch
                    approx_centroid = np.mean(quad, axis=0)
                    # find closest patches is only in 2d so then need to check depth
                    nearest_patches_ids = faults.find_closest_patches(approx_centroid[0], approx_centroid[1])
                    nearest_patches = [faults.patch_dic[patch_id] for patch_id in nearest_patches_ids]
                    min_z_diff = min([np.abs(patch.centre[2] - approx_centroid[2]) for patch in nearest_patches])
                    nearest_patches_z = [patch for patch in nearest_patches if
                                         np.abs(patch.centre[2] - approx_centroid[2]) == min_z_diff]
                    patch_ids = [patch.patch_number for patch in nearest_patches_z]

                    # find associated slip
                    slip = 0.
                    for patch_id in patch_ids:
                        if patch_id in self.patch_numbers:
                            slip_index = np.searchsorted(self.patch_numbers, patch_id)
                            slip_mag = self.patch_slip[slip_index]
                            slip += slip_mag
                    mean_slip = slip / len(patch_ids)
                    # check slip isn't 0/ less than a mm
                    if mean_slip > 1.e-3:
                        poly_list.append(Polygon(quad))

            except:
                print(f"{fName} could not be discretised - event slip distribution will be incomplete")

            # set parameters for OQ
            # parameters for openquake
            if hypocentre is None:
                hypocentre = np.array([self.x, self.y, self.z])
                print(f"Check hypocentre: {hypocentre}")
            evname = f'event_{self.event_id}_OQ'
            event_asOQ = OpenQuakeMultiSquareRupture(tile_list=poly_list, probability=probability, magnitude=self.mw,
                                                     rake=self.mean_rake, hypocentre=hypocentre, event_id=self.event_id,
                                                     name=evname, tectonic_region=tectonic_region,nztm2wgs=nztm2wgs)
            event_asOQ.to_oq_xml(write=os.path.join(outdir, f'event_{self.event_id}.xml'))

            return

    def slip_dist_quads_ktree(self, fault_model: RsqSimMultiFault, quads_dict: dict, min_moment: float = 1.e+18,
                              min_slip: float = 0.,threshold_for_inclusion: float = 0.5, slip_per_quad: bool = False):
        """
        Find ruptured quadrilateral tiles for each fault using a KD-tree.

        For each fault segment that exceeds ``min_moment``, uses a KD-tree
        on pre-computed quad centroids to assign triangular patches to
        quads, then returns quads where the fraction of ruptured patches
        exceeds ``threshold_for_inclusion``.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch dictionaries and patch centres.
        quads_dict : dict
            Mapping of fault segment name to a ``(n, 4, 3)`` array of
            quad corner coordinates (NZTM, m).
        min_moment : float, optional
            Minimum moment (N·m) for a fault to be considered.  Defaults
            to 1×10¹⁸ N·m.
        min_slip : float, optional
            Minimum patch slip (m) for a patch to count as ruptured.
            Defaults to 0.
        threshold_for_inclusion : float, optional
            Fraction of a quad's assigned patches that must be ruptured
            for the quad to be included.  Defaults to 0.5.
        slip_per_quad : bool, optional
            If ``True``, compute and store average slip per quad (not
            yet returned in output).

        Returns
        -------
        dict
            Mapping of fault segment name to a ``(n_ruptured, 4, 3)``
            array of ruptured quad corner coordinates.
        """
        moment_dict = self.make_fault_moment_dict(fault_model=fault_model, min_m0=min_moment, by_cfm_names=False)
        moment_quads = [key for key in moment_dict.keys() if key in quads_dict.keys()]
        missing_quads = [key for key in moment_dict.keys() if key not in moment_quads]
        if missing_quads:
            print("Warning: some fault segments have no associated quads")
            print(missing_quads)

        fault_patches = np.array(list(fault_model.patch_dic.keys()))
        ruptured_quads_dict = {}
        for name in moment_quads:
            segment_quads = quads_dict[name]
            if segment_quads.size > 0:
                segment = fault_model.name_dic[name]
                segment_quad_centroids = segment_quads.mean(axis=1)
                ruptured_patch_numbers = self.patch_numbers[np.isin(self.patch_numbers, fault_patches) & (self.patch_slip > min_slip)]
                segment_patch_centroids = segment.get_patch_centres()
                ruptured_patch_centroids = segment_patch_centroids[np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                tree = KDTree(segment_quad_centroids)
                _, all_patch_indices = tree.query(segment_patch_centroids)
                _, ruptured_patch_indices = tree.query(ruptured_patch_centroids)
                ruptured_quads = []
                quad_slip = []
                for i, quad in enumerate(segment_quads):
                    num_triangles = (all_patch_indices == i).sum()
                    num_ruptured_triangles = (ruptured_patch_indices == i).sum()
                    if num_ruptured_triangles / num_triangles > threshold_for_inclusion:
                        ruptured_quads.append(quad)
                        if slip_per_quad:
                            areas = np.array([segment.patch_dic[patch_number].area for patch_number in ruptured_patch_indices])
                            slip = self.patch_slip[np.isin(self.patch_numbers, ruptured_patch_indices)]
                            average_slip = np.average(slip, weights=areas)
                            quad_slip.append(average_slip)

                ruptured_quads_dict[name] = np.array(ruptured_quads)

        return ruptured_quads_dict

    def to_oq_points(self, fault_model: RsqSimMultiFault):
        """
        Convert event to OpenQuake point-source representation.

        Not yet implemented.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model (reserved for future use).
        """
        pass

    def get_crustal_component(self, fault_model: RsqSimMultiFault, crustal_names: list, min_moment: float = 1.e+18,
                              min_slip: float = 0.):
        """
        Compute moment, mean rake, and hypocentre for crustal fault components.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch dictionaries and patch moments.
        crustal_names : list of str
            Names of fault segments to treat as crustal.
        min_moment : float, optional
            Minimum moment (N·m) for a fault to be considered.
            Defaults to 1×10¹⁸ N·m.
        min_slip : float, optional
            Minimum patch slip (m) to count as ruptured.  Defaults to 0.

        Returns
        -------
        tuple of (float, float, numpy.ndarray) or None
            ``(mw, mean_rake, hypocentre_xyz)`` for the crustal
            component, or ``None`` if no crustal faults qualify.
        """
        moment_dict = self.make_fault_moment_dict(fault_model=fault_model, min_m0=min_moment, by_cfm_names=False)
        if any([name in crustal_names for name in moment_dict.keys()]):
            crustal_moment = 0.
            fault_patches = np.array(list(fault_model.patch_dic.keys()))
            rake_list = []
            moment_list = []
            hypocentre_time = 1.e20
            hypocentre = None
            for name, moment in moment_dict.items():
                if name in crustal_names:
                    crustal_moment += moment
                    segment = fault_model.name_dic[name]
                    ruptured_patch_numbers = self.patch_numbers[
                    np.isin(self.patch_numbers, fault_patches) & (self.patch_slip > min_slip)]
                    rakes = segment.rake[np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                    rake_list.append(rakes)
                    patch_moment = segment.patch_moments[np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                    moment_list.append(patch_moment)
                    patch_numbers_this_segment = segment.patch_numbers[
                        np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                    patch_times_this_segment = self.patch_time[np.isin(self.patch_numbers, patch_numbers_this_segment)]
                    segment_first_patch = fault_model.patch_dic[
                        patch_numbers_this_segment[np.argmin(patch_times_this_segment)]]
                    if patch_times_this_segment.min() < hypocentre_time:
                        hypocentre_time = patch_times_this_segment.min()
                        hypocentre = segment_first_patch.centre

            patch_moment_array = np.hstack(moment_list)
            rake_array = np.hstack(rake_list)
            mean_rake = weighted_circular_mean(rake_array, patch_moment_array)
            crustal_mw = m0_to_mw(crustal_moment)
            crustal_hypocentre = hypocentre

            hyp_x, hyp_y, hyp_z = crustal_hypocentre

            return crustal_mw, mean_rake, np.array([hyp_x, hyp_y, hyp_z])
        else:
            return None

    def get_subduction_component(self, fault_model: RsqSimMultiFault, subduction_names: list, min_moment: float = 1.e+18,
                                    min_slip: float = 0.):
        """
        Compute moment, mean rake, and hypocentre for subduction fault components.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model providing patch dictionaries and patch moments.
        subduction_names : list of str
            Names of fault segments to treat as subduction interface.
        min_moment : float, optional
            Minimum moment (N·m) for a fault to be considered.
            Defaults to 1×10¹⁸ N·m.
        min_slip : float, optional
            Minimum patch slip (m) to count as ruptured.  Defaults to 0.

        Returns
        -------
        tuple of (float, float, numpy.ndarray) or None
            ``(mw, mean_rake, hypocentre_xyz)`` for the subduction
            component, or ``None`` if no subduction faults qualify.
        """
        moment_dict = self.make_fault_moment_dict(fault_model=fault_model, min_m0=min_moment, by_cfm_names=False)
        if any([name in subduction_names for name in moment_dict.keys()]):
            subduction_moment = 0.
            fault_patches = np.array(list(fault_model.patch_dic.keys()))
            rake_list = []
            moment_list = []
            hypocentre_time = 1.e20
            hypocentre = None
            for name, moment in moment_dict.items():
                if name in subduction_names:
                    subduction_moment += moment
                    segment = fault_model.name_dic[name]
                    ruptured_patch_numbers = self.patch_numbers[
                    np.isin(self.patch_numbers, fault_patches) & (self.patch_slip > min_slip)]
                    rakes = segment.rake[np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                    rake_list.append(rakes)
                    patch_moment = segment.patch_moments[np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                    moment_list.append(patch_moment)
                    patch_numbers_this_segment = segment.patch_numbers[np.isin(segment.patch_numbers, ruptured_patch_numbers)]
                    patch_times_this_segment = self.patch_time[np.isin(self.patch_numbers, patch_numbers_this_segment)]
                    segment_first_patch = fault_model.patch_dic[patch_numbers_this_segment[np.argmin(patch_times_this_segment)]]
                    if patch_times_this_segment.min() < hypocentre_time:
                        hypocentre_time = patch_times_this_segment.min()
                        hypocentre = segment_first_patch.centre
            patch_moment_array = np.hstack(moment_list)
            rake_array = np.hstack(rake_list)
            mean_rake = weighted_circular_mean(rake_array, patch_moment_array)
            subduction_mw = m0_to_mw(subduction_moment)
            subduction_hypocentre = hypocentre

            hyp_x, hyp_y, hyp_z = subduction_hypocentre

            return subduction_mw, mean_rake, np.array([hyp_x, hyp_y, hyp_z])
        else:
            return None













    def slip_dist_to_quads(self, fault_model: RsqSimMultiFault, path2cfm: str, catalogue_version: str = 'v2',
                           vtk_dir: str = 'fault_vtks',
                           subd_tile_size: float = 15000., tile_size: float = 5000., ):
        """
        Write per-fault slip distributions as quadrilateral VTK meshes.

        Discretises each involved fault into rectangular tiles, assigns
        the average slip of the nearest triangular patches to each tile,
        and writes a VTK file per fault to
        ``<vtk_dir>/event_<id>_quad/<fault_name>_<event_id>.vtk``.

        Parameters
        ----------
        fault_model : RsqSimMultiFault
            Fault model for segment merging and discretisation.
        path2cfm : str
            Path to the Community Fault Model directory.
        catalogue_version : str, optional
            ``"v1"`` or ``"v2"``.  Defaults to ``"v2"``.
        vtk_dir : str, optional
            Output directory for VTK files.  Defaults to
            ``"fault_vtks"``.
        subd_tile_size : float, optional
            Tile size (m) for subduction faults.  Defaults to 15000 m.
        tile_size : float, optional
            Tile size (m) for crustal faults.  Defaults to 5000 m.
        """
        assert os.path.exists(path2cfm), "Path to CFM does not exist"
        if catalogue_version == 'v2':
            fault_model.make_v2_name_dic(path2cfm=path2cfm)
            name_dict = fault_model.v2_name_dic
        else:
            name_dict = dict(zip(fault_model.names, fault_model.names))
        unique_names = set(name_dict.values())

        # which faults are involved in this event?
        faults = RsqSimMultiFault(self.faults)
        faultNames = faults.names
        # find corresponding larger/cfm faults
        allFaults = np.unique([name_dict[name] for name in faultNames])
        subdFaults = np.unique([name_dict[name] for name in faultNames if
                                fnmatch.fnmatch(name, "*puysegar*") or fnmatch.fnmatch(name, "*hikurangi*")])

        outdir = os.path.join(vtk_dir, f'event_{self.event_id}_quad')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        poly_list = []
        for fName in allFaults:
            try:
                # print(fName)
                # need to find all parts of the fault, then later select those which have non-zero slip
                fault_merged = fault_model.merge_segments(fName, name_dict=name_dict, fault_name=fName)

                if len(subdFaults) > 0 and fName in subdFaults:
                    dip_angle = fault_merged.get_average_dip(subd_tile_size)
                    new_fault_rect = fault_merged.discretize_rectangular_tiles(tile_size=subd_tile_size)
                else:
                    # Average dip
                    dip_angle = fault_merged.get_average_dip()
                    # Discretize into rectangular tiles
                    new_fault_rect = fault_merged.discretize_rectangular_tiles(tile_size=tile_size)
                # Turn into quadrilateral mesh
                vertices = np.unique(np.vstack([rect for rect in new_fault_rect]), axis=0)
                vertex_dict = {tuple(vertex): i for i, vertex in enumerate(vertices)}
                new_fault_rect_indices = [[vertex_dict[tuple(vertex)] for vertex in quad] for quad in
                                          new_fault_rect]
                mesh = meshio.Mesh(points=vertices, cells={"quad": new_fault_rect_indices})

                # assign slip to these
                slip_list = []
                for quad in new_fault_rect:
                    # find nearest patch
                    approx_centroid = np.mean(quad, axis=0)
                    # find closest patches is only in 2d so then need to check depth
                    nearest_patches_ids = faults.find_closest_patches(approx_centroid[0], approx_centroid[1])
                    nearest_patches = [faults.patch_dic[patch_id] for patch_id in nearest_patches_ids]
                    min_z_diff = min([np.abs(patch.centre[2] - approx_centroid[2]) for patch in nearest_patches])
                    nearest_patches_z = [patch for patch in nearest_patches if
                                         np.abs(patch.centre[2] - approx_centroid[2]) == min_z_diff]
                    patch_ids = [patch.patch_number for patch in nearest_patches_z]

                    # find associated slip
                    slip = 0.
                    for patch_id in patch_ids:
                        if patch_id in self.patch_numbers:
                            slip_index = np.searchsorted(self.patch_numbers, patch_id)
                            slip_mag = self.patch_slip[slip_index]
                            slip += slip_mag
                    mean_slip = slip / len(patch_ids)
                    slip_list.append(mean_slip)
                mesh.cell_data = {"slip": np.array(slip_list)}
                meshio.write(os.path.join(outdir, f'{fName}_{self.event_id}.vtk'), mesh, file_format="vtk")
            except:
                print(f"{fName} could not be discretised - event slip distribution will be incomplete")
        return


class OpenQuakeMultiSquareRupture:
    """
    Multi-planar rupture representation for the OpenQuake engine.

    Packages a list of rectangular tile polygons (each representing a
    fault-surface patch) together with event metadata into the structure
    needed to write OpenQuake NRML ``multiPlanesRupture`` XML.

    Attributes
    ----------
    patches : list of OpenQuakeRectangularPatch
        Individual rectangular patches converted from input polygons.
    prob : float
        Probability of occurrence.
    magnitude : float
        Moment magnitude.
    rake : float
        Mean rake (degrees).
    hypocentre : numpy.ndarray
        Hypocentre coordinates ``[x, y, z]`` in the input CRS.
    inv_prob : float
        Complementary probability (``1 - prob``).
    hyp_depth : float
        Hypocentre depth in km (positive downward).
    hyp_lon, hyp_lat : float
        Hypocentre longitude and latitude in WGS84 degrees.
    event_id : int
        Catalogue event identifier.
    name : str
        Descriptive rupture name.
    tectonic_region : str
        Tectonic region tag for OpenQuake.
    """

    def __init__(self, tile_list: list[Polygon], probability: float, magnitude: float, rake: float,
                 hypocentre: np.ndarray, event_id: int, name: str = "Subduction earthquake",
                 tectonic_region: str = "subduction", nztm2wgs: bool = True):
        """
        Initialise an OpenQuake multi-planar rupture.

        Parameters
        ----------
        tile_list : list of Polygon
            Shapely polygons representing rectangular fault-surface tiles
            (in NZTM coordinates if ``nztm2wgs=True``).
        probability : float
            Probability of occurrence.
        magnitude : float
            Moment magnitude.
        rake : float
            Mean rake in degrees.
        hypocentre : numpy.ndarray
            ``[x, y, z]`` coordinates of the hypocentre.  If
            ``nztm2wgs=True``, ``x`` and ``y`` are NZTM eastings and
            northings (m); ``z`` is depth in m (negative downward).
        event_id : int
            Catalogue event identifier.
        name : str, optional
            Rupture name written to the XML.  Defaults to
            ``"Subduction earthquake"``.
        tectonic_region : str, optional
            Tectonic region tag.  Defaults to ``"subduction"``.
        nztm2wgs : bool, optional
            If ``True`` (default), transform hypocentre from NZTM to
            WGS84 longitude/latitude.
        """
        self.patches = [OpenQuakeRectangularPatch.from_polygon(tile) for tile in tile_list]
        self.prob = probability
        self.magnitude = magnitude
        self.rake = rake
        self.hypocentre = hypocentre
        self.inv_prob = 1. - probability

        self.hyp_depth = -1.e-3 * hypocentre[-1]
        if nztm2wgs:
            self.hyp_lon, self.hyp_lat = transformer_nztm2wgs.transform(hypocentre[0], hypocentre[1])
        else:
            self.hyp_lon, self.hyp_lat = hypocentre[0:2]
        self.event_id = event_id
        self.name = name
        self.tectonic_region = tectonic_region

    def to_oq_xml(self, write: str = None):
        """
        Build (and optionally write) an OpenQuake NRML rupture XML element.

        Constructs a ``<multiPlanesRupture>`` element containing magnitude,
        rake, hypocenter, and one ``<planarSurface>`` per patch.

        Parameters
        ----------
        write : str or None, optional
            Output file path.  If the path does not end with ``".xml"``
            the extension is appended automatically.  If ``None``, the
            XML is not written to disk.

        Returns
        -------
        xml.etree.ElementTree.Element
            The root ``<nrml>`` element (regardless of whether the file
            was written).
        """
        source_element = ElemTree.Element("nrml",
                                          attrib={"xmlns": "http://openquake.org/xmlns/nrml/0.4",
                                                  "xmlns:gml": "http://www.opengis.net/gml"
                                                  })
        multi_patch_elem = ElemTree.Element("multiPlanesRupture",
                                            attrib={"probs_occur": f"{self.inv_prob:.4f} {self.prob:.4f}"})
        mag_elem = ElemTree.Element("magnitude")
        mag_elem.text = f"{self.magnitude:.2f}"

        multi_patch_elem.append(mag_elem)
        rake_elem = ElemTree.Element("rake")
        rake_elem.text = f"{self.rake:.1f}"
        multi_patch_elem.append(rake_elem)

        hyp_element = ElemTree.Element("hypocenter", attrib={"depth": f"{self.hyp_depth:.3f}",
                                                             "lat": f"{self.hyp_lat:.4f}",
                                                             "lon": f"{self.hyp_lon:.4f}"})
        multi_patch_elem.append(hyp_element)
        for patch in self.patches:
            multi_patch_elem.append(patch.to_oq_xml())

        source_element.append(multi_patch_elem)

        if write is not None:
            assert isinstance(write, str)
            if write[-4:] != ".xml":
                write += ".xml"

            elmstr = ElemTree.tostring(source_element, encoding="UTF-8", method="xml")
            xml_dom = minidom.parseString(elmstr)
            pretty_xml_str = xml_dom.toprettyxml(indent="  ", encoding="utf-8")
            with open(write, "wb") as xml:
                xml.write(pretty_xml_str)

        return source_element
