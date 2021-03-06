from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast, plot_hillshade
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np
import pickle


def AnimateSequence(catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault, subduction_cmap: str = "plasma",
                    crustal_cmap: str = "viridis", global_max_slip: int = 10, global_max_sub_slip: int = 40,
                    step_size: int = 5, interval: int = 50, write: str = None, fps: int = 20, file_format: str = "gif",
                    figsize: tuple = (9.6, 7.2), hillshading_intensity: float = 0.0, bounds: tuple = None,
                    pickled_background : str = None, fading_increment: float = 2.0, plot_log: bool= False,
                    log_min: float = 1., log_max: float = 100., plot_subduction_cbar: bool = True,
                    plot_crustal_cbar: bool = True):
    """Shows an animation of a sequence of earthquake events over time

    Args:
        catalogue (RsqSimCatalogue): Catalogue of events to animate
        fault_model (RsqSimMultiFault): Fault model for events
        subduction_cmap (str): Colourmap for subduction colorbar
        crustal_cmap (str): Colourmap for crustal_cmap colorbar
        global_max_slip (int): Max slip to use for the colorscale
        global_max_sub_slip (int): Max subduction slip to use for the colorscale
        step_size (int): Step size to advance every interval
        interval (int): Time (ms) between each frame
        write (str): Write animation to file with given filename.
        fps (int): Frames per second.
        file_format (str): File extension for animation. Accepted values: gif, mp4, mov, avi.
        figsize (float, float): Width, height in inches.
        hillshading_intensity (float): Intensity of hillshading, value between 0-1.
    """
    assert file_format in ("gif", "mov", "avi", "mp4")

    # get all unique values
    event_list = np.unique(catalogue.event_list)
    # get RsqSimEvent objects
    events = catalogue.events_by_number(event_list.tolist(), fault_model)

    if pickled_background is not None:
        with open(pickled_background, "rb") as pfile:
            loaded_subplots = pickle.load(pfile)
        fig, coast_ax = loaded_subplots
    else:
        fig = plt.figure(figsize=figsize)

        # plot map
        coast_ax = fig.add_subplot(111, label="coast")
        if hillshading_intensity > 0:
            plot_coast(coast_ax, colors="0.0")
        else:
            plot_coast(coast_ax)

    coast_ax.set_aspect("equal")
    coast_ax.patch.set_alpha(0)
    coast_ax.get_xaxis().set_visible(False)
    coast_ax.get_yaxis().set_visible(False)

    num_events = len(events)
    all_plots = []
    timestamps = []
    for i, e in enumerate(events):
        plots = e.plot_slip_2d(
            subplots=(fig, coast_ax), global_max_slip=global_max_slip, global_max_sub_slip=global_max_sub_slip,
            bounds=bounds, plot_log_scale=plot_log, log_min=log_min, log_max=log_max)
        for p in plots:
            p.set_visible(False)
        years = math.floor(e.t0 / 3.154e7)
        all_plots.append(plots)
        timestamps.append(step_size * round(years/step_size))
        print("Plotting: " + str(i + 1) + "/" + str(num_events))

    if pickled_background is None:
        if hillshading_intensity > 0:
            x_lim = coast_ax.get_xlim()
            y_lim = coast_ax.get_ylim()
            plot_hillshade(coast_ax, hillshading_intensity)
            coast_ax.set_xlim(x_lim)
            coast_ax.set_ylim(y_lim)

    coast_ax_divider = make_axes_locatable(coast_ax)

    # Build colorbars
    if plot_log:
        log_ax = coast_ax_divider.append_axes("right", size="5%", pad=0.25)
        log_mappable = plots[0]
        log_cbar = fig.colorbar(
        log_mappable, cax=log_ax, extend='max')
        log_cbar.set_label("Slip (m)")

    else:
        sub_mappable = ScalarMappable(cmap=subduction_cmap)
        sub_mappable.set_clim(vmin=0, vmax=global_max_sub_slip)
        crust_mappable = ScalarMappable(cmap=crustal_cmap)
        crust_mappable.set_clim(vmin=0, vmax=global_max_slip)
        if plot_subduction_cbar:
            sub_ax = coast_ax_divider.append_axes("right", size="5%", pad=0.25)
            if plot_crustal_cbar:
                crust_ax = coast_ax_divider.append_axes("right", size="5%", pad=0.5)
            sub_cbar = fig.colorbar(
                sub_mappable, cax=sub_ax, extend='max')
            sub_cbar.set_label("Subduction slip (m)")
        else:
            crust_ax = coast_ax_divider.append_axes("right", size="5%", pad=0.25)
        crust_cbar = fig.colorbar(
            crust_mappable, cax=crust_ax, extend='max')
        crust_cbar.set_label("Slip (m)")

    # Slider to represent time progression
    axtime = coast_ax_divider.append_axes(
        "bottom", size="3%", pad=0.5)
    time_slider = Slider(
        axtime, 'Year', timestamps[0] - step_size, timestamps[-1] + step_size, valinit=timestamps[0] - step_size, valstep=step_size)

    axes = AxesSequence(fig, timestamps, all_plots, coast_ax, fading_increment=fading_increment)

    def update(val):
        time = time_slider.val
        axes.set_plot(time)
        if val == time_slider.valmax:
            axes.stop()
        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    def update_plot(num):
        val = time_slider.valmin + num * step_size
        time_slider.set_val(val)

    frames = int((time_slider.valmax - time_slider.valmin) / step_size) + 1
    animation = FuncAnimation(fig, update_plot,
                              interval=interval, frames=frames)

    if write is not None:
        writer = PillowWriter(fps=fps) if file_format == "gif" else FFMpegWriter(fps=fps)
        animation.save(f"{write}.{file_format}", writer)
    else:
        axes.show()


class AxesSequence(object):
    """Controls a series of plots on the screen and when they are visible"""

    def __init__(self, fig, timestamps, plots, coast_ax, fading_increment: float = 2.0):
        self.fig = fig
        self.timestamps = timestamps
        self.plots = plots
        self.coast_ax = coast_ax
        self.on_screen = []  # earthquakes currently displayed
        self._i = -1  # Currently displayed axes index
        self.fading_increment = fading_increment

    def set_plot(self, val):
        # plot corresponding event
        while self._i < len(self.timestamps) - 1 and val == self.timestamps[self._i + 1]:
            self._i += 1
            curr_plots = self.plots[self._i]
            for p in curr_plots:
                p.set_visible(True)
            self.on_screen.append(curr_plots)

        for i, p in enumerate(self.on_screen):
            self.fade(p, i)

    def fade(self, plot, index):
        visible = True
        for p in plot:
            opacity = p.get_alpha()
            if opacity / 2 <= 1e-2:
                p.set_alpha(1)
                visible = False
                p.set_visible(False)
            else:
                p.set_alpha(opacity / self.fading_increment)
        if not visible:
            self.on_screen.pop(index)

    def stop(self):
        for plot in self.on_screen:
            for p in plot:
                p.set_visible(False)
                p.set_alpha(1)
        self._i = -1
        self.on_screen.clear()

    def show(self):
        plt.show()
