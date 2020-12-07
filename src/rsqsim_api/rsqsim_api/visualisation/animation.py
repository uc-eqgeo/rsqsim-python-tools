from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
import os


def AnimateSequence(catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", global_max_slip: int = 0, step_size: int = 1e8, interval: int = 100):
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

    # Build colorbars
    sub_mappable = ScalarMappable(cmap=subduction_cmap)
    sub_mappable.set_clim(vmin=0, vmax=10)
    crust_mappable = ScalarMappable(cmap=crustal_cmap)
    crust_mappable.set_clim(vmin=0, vmax=10)
    sub_cbar = plt.colorbar(sub_mappable, ax=axes.fig.axes)
    sub_cbar.set_label("Subduction slip (m)")
    crust_cbar = plt.colorbar(crust_mappable, ax=axes.fig.axes)
    crust_cbar.set_label("Slip (m)")
    axes.sub_mappable = sub_mappable
    axes.crust_mappable = crust_mappable

    # Slider to represent time progression
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider = Slider(
        axtime, 'Time', axes.timestamps[0], axes.timestamps[-1], valstep=step_size)

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
        self.sub_mappable = None
        self.crust_mappable = None
        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        ax = self.fig.add_subplot(111, visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def set_plot(self, val):
        if val in self.timestamps:
            i = self.timestamps.index(val)
            self.axes[self._i].set_visible(False)
            self.axes[i].set_visible(True)
            self._i = i

    def show(self):
        plt.show()
