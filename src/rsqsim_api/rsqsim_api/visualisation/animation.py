from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
import math
import os


def AnimateSequence(catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", global_max_slip: int = 10, global_max_sub_slip: int = 40, step_size: int = 5, interval: int = 50):
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
    """

    # get all unique values
    event_list = dict.fromkeys(catalogue.event_list.tolist())
    # get RsqSimEvent objects
    events = catalogue.events_by_number(list(event_list), fault_model)
    axes = AxesSequence()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    num_events = len(events)
    for i, ax in zip(range(num_events), axes):
        max_slips = events[i].plot_slip_2d(
            show=False, show_coast=False, subplots=(axes.fig, ax), show_cbar=False, global_max_slip=global_max_slip, global_max_sub_slip=global_max_sub_slip)
        years = math.floor(events[i].t0 / 3.154e7)
        axes.timestamps.append(step_size * round(years/step_size))
        print("Plotting: " + str(i + 1) + "/" + str(num_events))

    # Build colorbars
    sub_mappable = ScalarMappable(cmap=subduction_cmap)
    sub_mappable.set_clim(vmin=0, vmax=global_max_sub_slip)
    crust_mappable = ScalarMappable(cmap=crustal_cmap)
    crust_mappable.set_clim(vmin=0, vmax=global_max_slip)
    sub_cbar = plt.colorbar(sub_mappable, ax=axes.fig.axes, extend='max')
    sub_cbar.set_label("Subduction slip (m)")
    crust_cbar = plt.colorbar(crust_mappable, ax=axes.fig.axes, extend='max')
    crust_cbar.set_label("Slip (m)")

    # Slider to represent time progression
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider = Slider(
        axtime, 'Year', axes.timestamps[0] - step_size, axes.timestamps[-1] + step_size, valinit=axes.timestamps[0] - step_size, valstep=step_size)

    def update(val):
        time = time_slider.val
        axes.set_plot(time)
        axes.fig.canvas.draw_idle()

    time_slider.on_changed(update)

    def update_plot(num):
        val = (time_slider.val + step_size - time_slider.valmin) % (
            time_slider.valmax - time_slider.valmin) + time_slider.valmin
        if val == time_slider.valmin:
            for ax in axes.on_screen:
                ax.set_visible(False)
                for obj in ax.findobj(match=PolyCollection):
                    obj.set_alpha(1)
            axes.on_screen.clear()
        time_slider.set_val(val)

    animation = FuncAnimation(axes.fig, update_plot, interval=interval)
    axes.show()


class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any given time."""

    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self.timestamps = []
        self.on_screen = []  # earthquakes currently displayed
        self.coast_ax = self.fig.add_subplot(111, label="coast")
        plot_coast(self.coast_ax)
        self.coast_ax.set_aspect("equal")
        self.coast_ax.patch.set_alpha(0)
        self.coast_ax.get_xaxis().set_visible(False)
        self.coast_ax.get_yaxis().set_visible(False)
        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        ax = self.fig.add_subplot(
            111, visible=False, label=self._n, sharex=self.coast_ax, sharey=self.coast_ax)
        ax.patch.set_alpha(0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
        self._n += 1
        self.axes.append(ax)
        return ax

    def set_plot(self, val):
        while val == self.timestamps[self._i + 1]:
            self._i += 1
            curr_ax = self.axes[self._i]
            curr_ax.set_visible(True)
            self.on_screen.append(curr_ax)
            if self._i == len(self.timestamps) - 1:
                self._i = -1  # ready for next loop

        for ax in self.on_screen:
            self.fade(ax)

    def fade(self, ax):
        visible = True
        for obj in ax.findobj(match=PolyCollection):
            opacity = obj.get_alpha()
            if opacity - .15 <= 0:
                obj.set_alpha(1)
                visible = False
            else:
                obj.set_alpha(opacity - .15)
        if visible is False:
            self.on_screen.remove(ax)
            ax.set_visible(False)

    def show(self):
        self.axes[self._i].set_visible(True)
        self.on_screen.append(self.axes[self._i])
        plt.show()
