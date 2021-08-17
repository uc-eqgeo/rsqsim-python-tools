from typing import Union
from collections import defaultdict
import pickle

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator
import numpy as np
from shapely.geometry import box

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast, plot_hillshade, plot_hillshade_niwa, plot_lake_polygons, plot_river_lines, plot_highway_lines, plot_boundary_polygons
from rsqsim_api.io.bruce_shaw_utilities import bruce_subduction


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
    def boundary(self):
        x1 = min([min(fault.vertices[:, 0]) for fault in self.faults])
        y1 = min([min(fault.vertices[:, 1]) for fault in self.faults])
        x2 = max([max(fault.vertices[:, 0]) for fault in self.faults])
        y2 = max([max(fault.vertices[:, 1]) for fault in self.faults])
        return [x1, y1, x2, y2]

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

        if event.patch_numbers.size > 0:
            patchnum_lookup = operator.itemgetter(*(event.patch_numbers))
            event.patches = list(patchnum_lookup(fault_model.patch_dic))
            event.faults = list(set(patchnum_lookup(fault_model.faults_with_patches)))
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


    def plot_background(self, figsize: tuple = (6.4, 4.8), hillshading_intensity: float = 0.0, bounds: tuple = None,
                        plot_rivers: bool = True, plot_lakes: bool = True, hillshade_fine: bool = False,
                        plot_highways: bool = True, plot_boundaries: bool = False, subplots=None,
                        pickle_name: str = None, hillshade_cmap: colors.LinearSegmentedColormap = cm.terrain):

        if subplots is not None:
            fig, ax = subplots
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(figsize)

        if bounds is not None:
            plot_bounds = list(bounds)
        else:
            plot_bounds = self.boundary

        if hillshading_intensity > 0:
            plot_coast(ax, clip_boundary=plot_bounds, colors="0.0")
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            if hillshade_fine:
                plot_hillshade_niwa(ax, hillshading_intensity, clip_bounds=plot_bounds, cmap=hillshade_cmap)
            else:
                plot_hillshade(ax, hillshading_intensity, clip_bounds=plot_bounds, cmap=hillshade_cmap)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
        else:
            plot_coast(ax, clip_boundary=plot_bounds)

        if plot_lakes:
            plot_lake_polygons(ax=ax, clip_bounds=plot_bounds)

        if plot_rivers:
            plot_river_lines(ax, clip_bounds=plot_bounds)

        if plot_highways:
            plot_highway_lines(ax, clip_bounds=plot_bounds)

        if plot_boundaries:
            plot_boundary_polygons(ax, clip_bounds=plot_bounds)

        ax.set_aspect("equal")
        x_lim = (plot_bounds[0], plot_bounds[2])
        y_lim = (plot_bounds[1], plot_bounds[3])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.set_xticks([])
        ax.set_yticks([])

        if pickle_name is not None:
            with open(pickle_name, "wb") as pfile:
                pickle.dump((fig, ax), pfile)

        return fig, ax

    def plot_slip_2d(self, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", show: bool = True,
                     write: str = None, subplots = None, global_max_sub_slip: int = 0, global_max_slip: int = 0,
                     figsize: tuple = (6.4, 4.8), hillshading_intensity: float = 0.0, bounds: tuple = None,
                     plot_rivers: bool = True, plot_lakes: bool = True,
                     plot_highways: bool = True, plot_boundaries: bool = False, create_background: bool = False,
                     coast_only: bool = True, hillshade_cmap: colors.LinearSegmentedColormap = cm.terrain):
        # TODO: Plot coast (and major rivers?)
        assert self.patches is not None, "Need to populate object with patches!"

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
            fig, ax = self.plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
                                           bounds=bounds, plot_rivers=plot_rivers, plot_lakes=plot_lakes,
                                           plot_highways=plot_highways, plot_boundaries=plot_boundaries,
                                           hillshade_cmap=hillshade_cmap)
        elif coast_only:
            fig, ax = self.plot_background(figsize=figsize, hillshading_intensity=hillshading_intensity,
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
                colours = np.zeros(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.patch_numbers:
                        slip_index = np.argwhere(self.patch_numbers == patch_id)[0]
                        colours[local_id] = self.patch_slip[slip_index]
                colour_dic[f_i] = colours
                if max(colours) > max_slip:
                    max_slip = max(colours)
        max_slip = global_max_sub_slip if global_max_sub_slip > 0 else max_slip

        plots = []

        # Plot subduction interface
        subduction_list = []
        subduction_plot = None
        for f_i, fault in enumerate(self.faults):
            if fault.name in bruce_subduction:
                subduction_list.append(fault.name)
                subduction_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                               facecolors=colour_dic[f_i],
                                               cmap=subduction_cmap, vmin=0, vmax=max_slip)
                plots.append(subduction_plot)

        max_slip = 0
        colour_dic = {}
        for f_i, fault in enumerate(self.faults):
            if fault.name not in bruce_subduction:
                colours = np.zeros(fault.patch_numbers.shape)
                for local_id, patch_id in enumerate(fault.patch_numbers):
                    if patch_id in self.patch_numbers:
                        slip_index = np.argwhere(self.patch_numbers == patch_id)[0]
                        colours[local_id] = self.patch_slip[slip_index]
                colour_dic[f_i] = colours
                if max(colours) > max_slip:
                    max_slip = max(colours)
        max_slip = global_max_slip if global_max_slip > 0 else max_slip

        crustal_plot = None
        for f_i, fault in enumerate(self.faults):
            if fault.name not in bruce_subduction:
                crustal_plot = ax.tripcolor(fault.vertices[:, 0], fault.vertices[:, 1], fault.triangles,
                                            facecolors=colour_dic[f_i],
                                            cmap=crustal_cmap, vmin=0, vmax=max_slip)
                plots.append(crustal_plot)

        if subplots is None:
            if subduction_list:
                sub_cbar = fig.colorbar(subduction_plot, ax=ax)
                sub_cbar.set_label("Subduction slip (m)")
            if crustal_plot is not None:
                crust_cbar = fig.colorbar(crustal_plot, ax=ax)
                crust_cbar.set_label("Slip (m)")


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
        plot_coast(ax, clip_boundary=self.boundary)
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

    def discretize_openquake(self):
        # Find faults
        pass




