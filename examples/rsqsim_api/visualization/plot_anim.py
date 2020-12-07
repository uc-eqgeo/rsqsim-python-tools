from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
import os


def main():
    run_dir = os.path.dirname(__file__)

    catalogue = RsqSimCatalogue.from_csv_and_arrays(
        os.path.join(run_dir, "../../../data/bruce_m7/bruce_m7_10kyr"))
    bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                          os.path.join(
        run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
        transform_from_utm=True)

    print(catalogue.catalogue_df)
    events = list(dict.fromkeys(catalogue.event_list.tolist()))
    print(events)
    events = catalogue.events_by_number(
        list(dict.fromkeys(catalogue.event_list.tolist()))[:50], bruce_faults)

    axes = AxesSequence()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    for i, ax in zip(range(len(events)), axes):
        max_slips = events[i].plot_slip_2d(
            show=False, clip=False, subplots=(axes.fig, ax), show_cbar=False, global_max_slip=10)
        axes.timestamps.append(round(events[i].t0, -8))
        print(i)

    sub_mappable = ScalarMappable(cmap="plasma")
    sub_mappable.set_clim(vmin=0, vmax=10)
    crust_mappable = ScalarMappable(cmap="viridis")
    crust_mappable.set_clim(vmin=0, vmax=10)
    sub_cbar = plt.colorbar(sub_mappable, ax=axes.fig.axes)
    sub_cbar.set_label("Subduction slip (m)")
    crust_cbar = plt.colorbar(crust_mappable, ax=axes.fig.axes)
    crust_cbar.set_label("Slip (m)")
    axes.sub_mappable = sub_mappable
    axes.crust_mappable = crust_mappable

    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    valstep = 1e8
    time_slider = Slider(axtime, 'Time', 25e8, 3200e8,
                         valinit=25e8, valstep=valstep)

    def update(val):
        time = time_slider.val
        axes.set_plot(time)
        axes.fig.canvas.draw_idle()

    time_slider.on_changed(update)

    def update_plot(num):
        val = (time_slider.val + valstep) % time_slider.valmax
        time_slider.set_val(val)

    animation = FuncAnimation(axes.fig, update_plot, interval=100)
    axes.show()


class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""

    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self.timestamps = []
        self.global_max_slip = 10
        self.coast_ax = self.fig.add_subplot(111, visible=False, label='coast')
        plot_coast(self.coast_ax)
        self.coast_ax.set_aspect("equal")
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
            self.coast_ax.set_visible(False)
            self._i = i

    def show(self):
        self.coast_ax.set_visible(True)
        plt.show()


if __name__ == '__main__':
    main()
