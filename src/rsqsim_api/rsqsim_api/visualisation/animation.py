from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
import os


def AnimateSequence(catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", global_max_slip: int = 10, step_size: int = 1e8, interval: int = 100):
    """Shows an animation of a sequence of earthquake events over time

    Args:
        catalogue (RsqSimCatalogue): Catalogue of events to animate
        fault_model (RsqSimMultiFault): Fault model for events
        subduction_cmap (str): Colourmap for subduction colorbar
        crustal_cmap (str): Colourmap for crustal_cmap colorbar
        global_max_slip (int): Max slip to use for the colorscale
        step_size (int): Step size to advance every interval
        interval (int): How long each frame lasts
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
            show=False, clip=False, subplots=(axes.fig, ax), show_cbar=False, global_max_slip=global_max_slip)
        axes.timestamps.append(round(events[i].t0, -8))
        print("Plotting: " + str(i+1) + "/" + str(num_events))

    # Plot coast
    coast_ax = axes.fig.add_subplot(111, label="coast")
    plot_coast(coast_ax)
    coast_ax.set_aspect("equal")
    coast_ax.patch.set_alpha(0)

    # Build colorbars
    sub_mappable = ScalarMappable(cmap=subduction_cmap)
    sub_mappable.set_clim(vmin=0, vmax=global_max_slip)
    crust_mappable = ScalarMappable(cmap=crustal_cmap)
    crust_mappable.set_clim(vmin=0, vmax=global_max_slip)
    sub_cbar = plt.colorbar(sub_mappable, ax=axes.fig.axes, extend='max')
    sub_cbar.set_label("Subduction slip (m)")
    crust_cbar = plt.colorbar(crust_mappable, ax=axes.fig.axes, extend='max')
    crust_cbar.set_label("Slip (m)")
    axes.sub_mappable = sub_mappable
    axes.crust_mappable = crust_mappable

    # Slider to represent time progression
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider = Slider(
        axtime, 'Time', axes.timestamps[0], axes.timestamps[-1], valinit=axes.timestamps[0], valstep=step_size)

    def update(val):
        time = time_slider.val
        axes.set_plot(time)
        axes.fig.canvas.draw_idle()

    time_slider.on_changed(update)

    def update_plot(num):
        val = (time_slider.val + step_size) % time_slider.valmax
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
        self.sub_mappable = None
        self.crust_mappable = None
        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        ax = self.fig.add_subplot(111, visible=False, label=self._n)
        ax.patch.set_alpha(0)
        self._n += 1
        self.axes.append(ax)
        return ax

    def set_plot(self, val):
        if val in self.timestamps:
            i = self.timestamps.index(val)
            curr_ax = self.axes[i]
            curr_ax.set_visible(True)
            self.fade(curr_ax)
            self.on_screen.append(curr_ax)
            self._i = i
        for ax in self.on_screen:
            self.fade(ax)

    def fade(self, ax):
        visible = True
        for obj in ax.findobj(match=PolyCollection):
            opacity = obj.get_alpha()
            if opacity - .05 <= 0:
                obj.set_alpha(1)
                visible = False
            else:
                obj.set_alpha(opacity - .05)
        if visible is False:
            self.on_screen.remove(ax)
            ax.set_visible(False)

    def show(self):
        self.axes[0].set_visible(True)
        self.fade(self.axes[0])
        self.on_screen.append(self.axes[0])
        plt.show()
