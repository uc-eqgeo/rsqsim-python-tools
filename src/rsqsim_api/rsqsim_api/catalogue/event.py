from typing import Union, List
from collections import defaultdict
import pickle

import xml.etree.ElementTree as ElemTree
from xml.dom import minidom

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pyproj import Transformer
import geopandas as gpd

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast, plot_background
from rsqsim_api.io.bruce_shaw_utilities import bruce_subduction
from rsqsim_api.io.mesh_utils import array_to_mesh
from rsqsim_api.fault.patch import OpenQuakeRectangularPatch

transformer_nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)


class RsqSimEvent:
    def __init__(self):
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

    @property
    def num_faults(self):
        return len(self.faults)

    @property
    def bounds(self):
        x1 = min([min(fault.vertices[:, 0]) for fault in self.faults])
        y1 = min([min(fault.vertices[:, 1]) for fault in self.faults])
        x2 = max([max(fault.vertices[:, 0]) for fault in self.faults])
        y2 = max([max(fault.vertices[:, 1]) for fault in self.faults])
        return [x1, y1, x2, y2]

    @property
    def exterior(self):
        return unary_union([patch.as_polygon() for patch in self.patches])

    @classmethod
    def from_catalogue_array(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float, event_id: int = None):
        """

        :param t0:
        :param m0:
        :param mw:
        :param x:
        :param y:
        :param z:
        :param area:
        :param dt:
        :param event_id:
        :return:
        """

        event = cls()
        event.t0, event.m0, event.mw, event.x, event.y, event.z = [t0, m0, mw, x, y, z]
        event.area, event.dt = [area, dt]
        event.event_id = event_id

        return event

    @property
    def patch_outline_gs(self):
        return gpd.GeoSeries([patch.as_polygon() for patch in self.patches], crs=2193)

    @classmethod
    def from_earthquake_list(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: Union[list, np.ndarray, tuple],
                             patch_slip: Union[list, np.ndarray, tuple],
                             patch_time: Union[list, np.ndarray, tuple],
                             fault_model: RsqSimMultiFault, filter_single_patches: bool = True,
                             min_patches: int = 10, min_slip: Union[float, int] = 1, event_id: int = None):
        event = cls.from_catalogue_array(
            t0, m0, mw, x, y, z, area, dt, event_id=event_id)

        faults_with_patches = fault_model.faults_with_patches
        patches_on_fault = defaultdict(list)
        [ patches_on_fault[faults_with_patches[i]].append(i) for i in patch_numbers ]

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

        return event

    @classmethod
    def from_multiprocessing(cls, t0: float, m0: float, mw: float, x: float,
                             y: float, z: float, area: float, dt: float,
                             patch_numbers: Union[list, np.ndarray, tuple],
                             patch_slip: Union[list, np.ndarray, tuple],
                             patch_time: Union[list, np.ndarray, tuple],
                             fault_model: RsqSimMultiFault, mask: list, event_id: int = None):
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


    def plot_slip_2d(self, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", show: bool = True,
                     write: str = None, subplots = None, global_max_sub_slip: int = 0, global_max_slip: int = 0,
                     figsize: tuple = (6.4, 4.8), hillshading_intensity: float = 0.0, bounds: tuple = None,
                     plot_rivers: bool = True, plot_lakes: bool = True,
                     plot_highways: bool = True, plot_boundaries: bool = False, create_background: bool = False,
                     coast_only: bool = True, hillshade_cmap: colors.LinearSegmentedColormap = cm.terrain,
                     plot_log_scale: bool = False, log_cmap: str = "magma", log_min: float = 1.0,
                     log_max: float = 100., plot_traces: bool = True, trace_colour: str = "pink",
                     min_slip_percentile: float = None, min_slip_value: float = None, plot_zeros: bool = True):
        # TODO: Plot coast (and major rivers?)
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
                fig, ax = loaded_subplots
            else:
                # Assume matplotlib objects
                fig, ax = subplots
        elif create_background:
            fig, ax = plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                      bounds=bounds, plot_rivers=plot_rivers, plot_lakes=plot_lakes,
                                      plot_highways=plot_highways, plot_boundaries=plot_boundaries,
                                      hillshade_cmap=hillshade_cmap)
        elif coast_only:
            fig, ax = plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                      bounds=bounds, plot_rivers=False, plot_lakes=False, plot_highways=False,
                                      plot_boundaries=False, hillshade_cmap=hillshade_cmap)


        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(figsize)

        # Find maximum slip for subduction interface

        # Find maximum slip to scale colourbar
        max_slip = 0

        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name in bruce_subduction:
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
            if fault.name in bruce_subduction:
                subduction_list.append(fault.name)
                if plot_log_scale:
                    subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max))
                else:
                    subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=subduction_cmap, vmin=0., vmax=max_slip)
                plots.append(subduction_plot)

        max_slip = 0
        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name not in bruce_subduction:
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
        for f_i, fault in enumerate(self.faults):
            if fault.name not in bruce_subduction:
                if plot_log_scale:
                    crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                   facecolors=colour_dic[f_i],
                                                   cmap=log_cmap, norm=colors.LogNorm(vmin=log_min, vmax=log_max))
                else:
                    crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                                facecolors=colour_dic[f_i],
                                                cmap=crustal_cmap, vmin=0., vmax=max_slip)
                plots.append(crustal_plot)

        if any([subplots is None, isinstance(subplots,str)]):
            if plot_log_scale:
                if subduction_list:
                    sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                    sub_cbar.set_label("Slip (m)")
                elif crustal_plot is not None:
                    crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                    crust_cbar.set_label("Slip (m)")
            else:
                if subduction_list:
                    sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                    sub_cbar.set_label("Subduction slip (m)")
                if crustal_plot is not None:
                    crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                    crust_cbar.set_label("Slip (m)")



        plot_coast(ax=ax)


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
                            figsize: tuple = (6.4, 4.8)):

        assert file_format in ("gif", "mov", "avi", "mp4")

        fig, ax = plt.subplots()
        fig.set_size_inches(figsize)
        plot_coast(ax, clip_boundary=self.bounds)
        ax.set_aspect("equal")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        colour_dic = {}
        timestamps = defaultdict(set)
        subduction_max_slip = 0
        crustal_max_slip = 0
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
            if fault.name in bruce_subduction:
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
            if fault.name in bruce_subduction:
                subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                    facecolors=init_colours,
                                    cmap=subduction_cmap, vmin=0, vmax=subduction_max_slip)
                plots[f_i] = (subduction_plot, init_colours)

        crustal_plot = None
        for f_i, fault in enumerate(self.faults):
            init_colours = np.zeros(fault.patch_numbers.shape)
            if fault.name not in bruce_subduction:
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
            animation.save(f"{write}.{file_format}", writer)

        if show:
            plt.show()

    def slip_dist_array(self, include_zeros: bool = True, min_slip_percentile: float = None,
                        min_slip_value: float = None, nztm_to_lonlat: bool = False):
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

    def slip_dist_to_mesh(self, include_zeros: bool = True, min_slip_percentile: float = None,
                          min_slip_value: float = None, nztm_to_lonlat: bool = False):

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
        mesh = self.slip_dist_to_mesh(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                      min_slip_value=min_slip_value)
        mesh.write(vtk_file, file_format="vtk")

    def slip_dist_to_obj(self, obj_file: str, include_zeros: bool = True, min_slip_percentile: float = None,
                         min_slip_value: float = None):
        mesh = self.slip_dist_to_mesh(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                      min_slip_value=min_slip_value)
        mesh.write(obj_file, file_format="obj")

    def slip_dist_to_txt(self, txt_file, include_zeros: bool = True, min_slip_percentile: float = None,
                         min_slip_value: float = None, nztm_to_lonlat: bool = False):
        if nztm_to_lonlat:
            header="lon1 lat1 z1 lon2 lat2 z2 lon3 lat3 z3 slip_m rake_deg time_s"
        else:
            header = "x1 y1 z1 x2 y2 z2 x3 y3 z3 slip_m rake_deg time_s"
        slip_dist_array = self.slip_dist_array(include_zeros=include_zeros, min_slip_percentile=min_slip_percentile,
                                               min_slip_value=min_slip_value, nztm_to_lonlat=nztm_to_lonlat)
        np.savetxt(txt_file, slip_dist_array, fmt="%.6f", delimiter=" ", header=header)

    def discretize_openquake(self, tile_list: List[Polygon], probability: float, rake: float):
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




class OpenQuakeMultiSquareRupture:
    def __init__(self, tile_list: List[Polygon], probability: float, magnitude: float, rake: float,
                 hypocentre: np.ndarray, event_id: int, name: str = "Subduction earthquake",
                 tectonic_region: str = "subduction"):
        self.patches = [OpenQuakeRectangularPatch.from_polygon(tile) for tile in tile_list]
        self.prob = probability
        self.magnitude = magnitude
        self.rake = rake
        self.hypocentre = hypocentre
        self.inv_prob = 1. - probability

        self.hyp_depth = -1.e-3 * hypocentre[-1]
        self.hyp_lon, self.hyp_lat = transformer_nztm2wgs.transform(hypocentre[0], hypocentre[1])
        self.event_id = event_id
        self.name = name
        self.tectonic_region = tectonic_region

    def to_oq_xml(self, write: str = None):
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






