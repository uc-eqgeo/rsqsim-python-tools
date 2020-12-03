from PIL import Image
from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
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
    events = catalogue.events_by_number([88, 483, 499, 528, 588], bruce_faults)

    fig = plt.figure()
    coast_ax = fig.add_subplot(111, visible=False, label='coast')
    plot_coast(coast_ax)
    coast_ax.set_aspect("equal")
    axes = AxesSequence()
    axes.fig = fig
    axes.coast_ax = coast_ax
    plt.subplots_adjust(left=0.25, bottom=0.25)
    for i, ax in zip(range(len(events)), axes):
        events[i].plot_slip_2d(show=False, clip=False, subplots=(axes.fig, ax))
        axes.timestamps.append(round(events[i].t0, -8))

    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider = Slider(axtime, 'Time', 25e8, 100e8,
                         valinit=25e8, valstep=1e8)

    def update(val):
        time = time_slider.val
        axes.set_plot(time)
        axes.fig.canvas.draw_idle()

    time_slider.on_changed(update)
    axes.show()


class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""

    def __init__(self):
        self.fig = None
        self.axes = []
        self.timestamps = []
        self.coast_ax = None
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
        else:
            self.axes[self._i].set_visible(False)
            self.coast_ax.set_visible(True)

    def show(self):
        plt.show()


if __name__ == '__main__':
    main()
