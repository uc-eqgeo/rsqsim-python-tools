from PIL import Image
from rsqsim_api.containers.catalogue import RsqSimCatalogue
from rsqsim_api.containers.fault import RsqSimMultiFault
from matplotlib import pyplot as plt
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

    axes = AxesSequence()
    for i, ax in zip(range(len(events)), axes):
        events[i].plot_slip_2d(show=False, clip=False, subplots=(axes.fig, ax))
    axes.show()


class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""

    def __init__(self):
        self.fig = plt.figure()
        self.axes = []
        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        ax = self.fig.add_subplot(111, visible=False, label=self._n)
        self._n += 1
        self.axes.append(ax)
        return ax

    def on_keypress(self, event):
        if event.key == 'x':
            self.next_plot()
        elif event.key == 'z':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes):
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()


if __name__ == '__main__':
    main()
