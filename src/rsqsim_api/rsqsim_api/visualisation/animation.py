"""
Animation utilities for visualising earthquake sequences over time.

Provides :func:`AnimateSequence` for generating slip animations driven
by a :class:`~rsqsim_api.catalogue.catalogue.RsqSimCatalogue`,
:class:`AxesSequence` for managing per-event plot visibility and
fading, :func:`plot_axis_sequence` for driving the slider/animation
loop, and :func:`write_animation_frame` / :func:`write_animation_frames`
for parallel frame-by-frame rendering.
"""
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.visualisation.utilities import plot_coast, plot_hillshade
from rsqsim_api.io.rsqsim_constants import seconds_per_year
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor, wait, FIRST_COMPLETED
import os.path
import math
import numpy as np
import pickle
from io import BytesIO

from multiprocessing import Pool
from functools import partial


def AnimateSequence(catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault, subduction_cmap: str = "plasma",
                    crustal_cmap: str = "viridis", global_max_slip: int = 10, global_max_sub_slip: int = 40,
                    step_size: int = 1, interval: int = 50, write: str = None, fps: int = 20, file_format: str = "gif",
                    figsize: tuple = (9.6, 7.2), hillshading_intensity: float = 0.0, bounds: tuple = None,
                    pickled_background : str = None, fading_increment: float = 2.0, plot_log: bool= False,
                    log_min: float = 1., log_max: float = 100., plot_subduction_cbar: bool = True,
                    plot_crustal_cbar: bool = True, min_slip_value: float = None, plot_zeros: bool = True,
                    extra_sub_list: list = None, plot_cbars: bool = False, write_frames: bool = False,
                    pickle_plots: str = None, load_pickle_plots: str = None, num_threads: int = 4, **kwargs):
    """
    Show an animation of a sequence of earthquake events over time.

    Plots per-event slip distributions onto a NZ map background and
    animates them with a time slider.  Supports pre-rendered pickled
    backgrounds and frame-by-frame writing via
    :func:`plot_axis_sequence`.

    Parameters
    ----------
    catalogue : RsqSimCatalogue
        Catalogue of events to animate.
    fault_model : RsqSimMultiFault
        Fault model providing patch geometry for each event.
    subduction_cmap : str, optional
        Colourmap name for the subduction colourbar.
        Defaults to ``"plasma"``.
    crustal_cmap : str, optional
        Colourmap name for the crustal colourbar.
        Defaults to ``"viridis"``.
    global_max_slip : int, optional
        Maximum crustal slip (m) for the colour scale.  Defaults to 10.
    global_max_sub_slip : int, optional
        Maximum subduction slip (m) for the colour scale.
        Defaults to 40.
    step_size : int, optional
        Year increment per animation frame.  Defaults to 1.
    interval : int, optional
        Delay between frames in milliseconds.  Defaults to 50.
    write : str or None, optional
        Output file path (without extension).  If ``None``, display
        interactively.
    fps : int, optional
        Frames per second for the output file.  Defaults to 20.
    file_format : str, optional
        Output format: ``"gif"``, ``"mp4"``, ``"mov"``, or ``"avi"``.
        Defaults to ``"gif"``.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
        Defaults to ``(9.6, 7.2)``.
    hillshading_intensity : float, optional
        Hillshade overlay transparency (0–1).  Defaults to 0.
    bounds : tuple or None, optional
        ``(x_min, y_min, x_max, y_max)`` map extent.
    pickled_background : str or None, optional
        Path to a pickled ``(fig, ax)`` background.
    fading_increment : float, optional
        Fading divisor per time step.  Defaults to 2.0.
    plot_log : bool, optional
        If ``True``, use a log colour scale.  Defaults to ``False``.
    log_min : float, optional
        Lower bound for the log scale.  Defaults to 1.
    log_max : float, optional
        Upper bound for the log scale.  Defaults to 100.
    plot_subduction_cbar : bool, optional
        If ``True`` (default), show the subduction colourbar.
    plot_crustal_cbar : bool, optional
        If ``True`` (default), show the crustal colourbar.
    min_slip_value : float or None, optional
        Minimum slip to plot; patches below this are hidden.
    plot_zeros : bool, optional
        If ``True`` (default), plot patches with zero slip.
    extra_sub_list : list or None, optional
        Additional subduction patch numbers to highlight.
    plot_cbars : bool, optional
        If ``True``, plot per-event colourbars.  Defaults to ``False``.
    write_frames : bool, optional
        If ``True``, write individual PNG frames instead of animating.
        Defaults to ``False``.
    pickle_plots : str or None, optional
        Path to save pre-rendered plot pickles.
    load_pickle_plots : str or None, optional
        Path to load pre-rendered plot pickles.
    num_threads : int, optional
        Worker count for parallel frame rendering.  Defaults to 4.
    """
    assert file_format in ("gif", "mov", "avi", "mp4")

    # get all unique values
    event_list = np.unique(catalogue.event_list)
    # get RsqSimEvent objects
    events = catalogue.events_by_number(event_list.tolist(), fault_model)

    if pickled_background is not None:
        with open(pickled_background, "rb") as pfile:
            loaded_subplots = pickle.load(pfile)
        fig, background_ax = loaded_subplots
        coast_ax = background_ax["main_figure"]
        slider_ax = background_ax["slider"]
        year_ax = background_ax["year"]
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

    if pickled_background is None:
        if hillshading_intensity > 0:
            x_lim = coast_ax.get_xlim()
            y_lim = coast_ax.get_ylim()
            plot_hillshade(coast_ax, hillshading_intensity)
            coast_ax.set_xlim(x_lim)
            coast_ax.set_ylim(y_lim)

    num_events = len(events)

    conditions_for_load_pickle_plots = load_pickle_plots is not None and os.path.exists(load_pickle_plots)
    if conditions_for_load_pickle_plots:
        if os.path.exists(load_pickle_plots):
            with open(load_pickle_plots, "rb") as pfile:
                loaded_subplots = pickle.load(pfile)
            fig, background_ax, coast_ax, slider_ax, year_ax, all_plots, timestamps = loaded_subplots
            print("Loaded plots from pickle file")

    else:
        all_plots = []
        timestamps = []
        for i, e in enumerate(events):
            plots = e.plot_slip_2d(
                subplots=(fig, coast_ax), global_max_slip=global_max_slip, global_max_sub_slip=global_max_sub_slip,
                bounds=bounds, plot_log_scale=plot_log, log_min=log_min, log_max=log_max, min_slip_value=min_slip_value,
                plot_zeros=plot_zeros, extra_sub_list=extra_sub_list, plot_cbars=plot_cbars)
            for p in plots:
                p.set_visible(False)
            years = math.floor(e.t0 / (3.154e7))
            all_plots.append(plots)
            timestamps.append(step_size * round(years/step_size))
            print("Plotting: " + str(i + 1) + "/" + str(num_events))

        if pickle_plots is not None:
            with open(pickle_plots, "wb") as pfile:
                pickle.dump((fig, background_ax, coast_ax, slider_ax, year_ax, all_plots, timestamps), pfile)




    time_slider_all = Slider(
        slider_ax, 'Year', timestamps[0] - step_size, timestamps[-1] + step_size,
        valinit=timestamps[0] - step_size, valstep=step_size)
    frames = int((time_slider_all.valmax - time_slider_all.valmin) / step_size) + 1

    if num_threads > 1:
        split_frames = np.array_split(np.arange(frames), num_threads)
        arg_holder = []
        with ProcessPoolExecutor(max_workers=num_threads) as plot_executor:
            for i in range(num_threads):
                print(f"Starting thread {i}")
                with open(load_pickle_plots, "rb") as pfile:
                    loaded_subplots = pickle.load(pfile)
                arg_holder.append(loaded_subplots)
                fig, background_ax, coast_ax, slider_ax, year_ax, all_plots_i, timestamps_i = arg_holder[i]
                pickled_figure = fig, background_ax, coast_ax, slider_ax, year_ax
                pool_kwargs = { "step_size": step_size, "interval": interval,
                                "write": write, "write_frames": write_frames, "file_format": file_format, "fps": fps,
                                "fading_increment": fading_increment, "figsize": figsize,
                                "hillshading_intensity": hillshading_intensity}


                plot_executor.submit(plot_axis_sequence, split_frames[i], timestamps_i, all_plots_i, pickled_figure,
                                     **pool_kwargs)

    else:
        pickled_figure = fig, background_ax, coast_ax, slider_ax, year_ax
        plot_axis_sequence(frames, pickled_background=pickled_figure, timestamps=timestamps, all_plots=all_plots,
                            step_size=step_size, interval=interval, write=write, write_frames=write_frames,
                            file_format=file_format, fps=fps, fading_increment=fading_increment, figsize=figsize,
                            hillshading_intensity=hillshading_intensity)



class AxesSequence(object):
    """
    Manage the visibility and fading of a time-ordered sequence of plots.

    Tracks which event plots are currently on screen and progressively
    fades them out according to ``fading_increment`` as the animation
    advances.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure containing all plots.
    timestamps : list of int
        Sorted year timestamps corresponding to each entry in ``plots``.
    plots : list of list
        Per-event lists of matplotlib artist objects.
    coast_ax : matplotlib.axes.Axes
        The main map axis.
    fading_increment : float
        Alpha divisor applied each time step.  Defaults to 2.0.
    on_screen : list
        Currently visible plot groups.
    """

    def __init__(self, fig, timestamps, plots, coast_ax, fading_increment: float = 2.0):
        self.fig = fig
        self.timestamps = timestamps
        self.plots = plots
        self.coast_ax = coast_ax
        self.on_screen = []  # earthquakes currently displayed
        self._i = -1  # Currently displayed axes index
        self.fading_increment = fading_increment

    def set_plot(self, val):
        """
        Advance the sequence to show all events at time ``val`` and fade older ones.

        Parameters
        ----------
        val : int
            Current slider year value.
        """
        # plot corresponding event
        while self._i < len(self.timestamps) - 1 and val == self.timestamps[self._i + 1]:
            self._i += 1
            curr_plots = self.plots[self._i]
            print(curr_plots)
            for p in curr_plots:
                p.set_visible(True)
            self.on_screen.append(curr_plots)
            print(self.on_screen)

        for i, p in enumerate(self.on_screen):
            self.fade(p, i)

    def fade(self, plot, index):
        """
        Reduce the alpha of a plot group and hide it once fully transparent.

        Parameters
        ----------
        plot : list
            List of matplotlib artists for one event.
        index : int
            Position of ``plot`` in :attr:`on_screen`; removed if invisible.
        """
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
        """Hide all on-screen plots and reset the sequence to the start."""
        for plot in self.on_screen:
            for p in plot:
                p.set_visible(False)
                p.set_alpha(1)
        self._i = -1
        self.on_screen.clear()

    def show(self):
        """Display the animation figure interactively."""
        plt.show()


def plot_axis_sequence(frames, timestamps, all_plots, pickled_background, step_size=1,
                       interval=50, write=None, write_frames=False, file_format="gif", fps=20, fading_increment=2.0,
                       figsize: tuple = (9.6, 7.2), hillshading_intensity: float = 0.0):
    """
    Drive the slider animation loop for a pre-rendered set of event plots.

    Attaches an :class:`AxesSequence` to a time slider and either
    saves individual frames, saves an animation file, or shows the
    animation interactively.

    Parameters
    ----------
    frames : int or array-like
        Number of frames, or array of frame indices.
    timestamps : list of int
        Year timestamps for each entry in ``all_plots``.
    all_plots : list of list
        Per-event lists of matplotlib artists.
    pickled_background : tuple or None
        ``(fig, background_ax, coast_ax, slider_ax, year_ax)`` tuple
        loaded from a pickled background, or ``None`` to build one.
    step_size : int, optional
        Year increment per frame.  Defaults to 1.
    interval : int, optional
        Delay between frames in milliseconds.  Defaults to 50.
    write : str or None, optional
        Output file path (without extension).  If ``None``, show
        interactively.
    write_frames : bool, optional
        If ``True``, write individual PNG frames to ``frames/``.
        Defaults to ``False``.
    file_format : str, optional
        Output format: ``"gif"``, ``"mp4"``, ``"mov"``, or ``"avi"``.
        Defaults to ``"gif"``.
    fps : int, optional
        Frames per second for the output file.  Defaults to 20.
    fading_increment : float, optional
        Alpha divisor per time step.  Defaults to 2.0.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.
        Defaults to ``(9.6, 7.2)``.
    hillshading_intensity : float, optional
        Hillshade transparency (0–1).  Defaults to 0.
    """

    if pickled_background is not None:

        fig, background_ax, coast_ax, slider_ax, year_ax = pickled_background
        coast_ax = background_ax["main_figure"]
        slider_ax = background_ax["slider"]
        year_ax = background_ax["year"]
        year_text = year_ax.text(0.5, 0.5, str(int(0)), horizontalalignment='center', verticalalignment='center',
                                 fontsize=12)
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

    if pickled_background is None:
        if hillshading_intensity > 0:
            x_lim = coast_ax.get_xlim()
            y_lim = coast_ax.get_ylim()
            plot_hillshade(coast_ax, hillshading_intensity)
            coast_ax.set_xlim(x_lim)
            coast_ax.set_ylim(y_lim)



    time_slider = Slider(
        slider_ax, 'Year', timestamps[0] - step_size, timestamps[-1] + step_size, valinit=timestamps[0] - step_size, valstep=step_size)
    time_slider.valtext.set_visible(False)

    axes = AxesSequence(fig, timestamps, all_plots, coast_ax, fading_increment=fading_increment)
    print(all_plots)

    def update(val):
        time = time_slider.val
        axes.set_plot(time)
        if val == time_slider.valmax:
            axes.stop()
        year_text.set_text(str(int(time)))
        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    def update_plot(num):
        val = time_slider.valmin + num * step_size
        time_slider.set_val(val)

    if write_frames:
        for i in range(frames):
            update_plot(i)
            fig.savefig(f"frames/frame{i:04d}.png", dpi=300)
    else:
        animation = FuncAnimation(fig, update_plot,
                                  interval=interval, frames=frames)

        if write is not None:
            writer = PillowWriter(fps=fps) if file_format == "gif" else FFMpegWriter(fps=fps)
            animation.save(f"{write}.{file_format}", writer, dpi=300)
        else:
            axes.show()


def write_animation_frame(frame_num, frame_time, start_time, end_time, step_size, catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault,
                          pickled_background: str,
                           subduction_cmap: str = "plasma", crustal_cmap: str = "viridis", global_max_slip: int = 10,
                           global_max_sub_slip: int = 40,
                           bounds: tuple = None, fading_increment: float = 2.0, time_to_threshold: float = 10.,
                           plot_log: bool = False, log_min: float = 1., log_max: float = 100.,
                           min_slip_value: float = None, plot_zeros: bool = True, extra_sub_list: list = None,
                           min_mw: float = None, decimals: int = 1, subplot_name: str = "main_figure"):
    """
    Render a single animation frame and return the figure.

    Filters the catalogue to events within ``time_to_threshold`` years
    before ``frame_time``, plots their slip distributions with faded
    alpha, and returns the figure for saving.

    Parameters
    ----------
    frame_num : int
        Frame index (used as the return key).
    frame_time : float
        Current animation time in years.
    start_time : float
        Animation start time in years.
    end_time : float
        Animation end time in years.
    step_size : int
        Year increment per frame.
    catalogue : RsqSimCatalogue
        Event catalogue to filter.
    fault_model : RsqSimMultiFault
        Fault model for plotting slip distributions.
    pickled_background : str
        Path to a pickled ``(fig, ax)`` background file.
    subduction_cmap : str, optional
        Colourmap for subduction slip.  Defaults to ``"plasma"``.
    crustal_cmap : str, optional
        Colourmap for crustal slip.  Defaults to ``"viridis"``.
    global_max_slip : int, optional
        Maximum crustal slip (m) for the colour scale.  Defaults to 10.
    global_max_sub_slip : int, optional
        Maximum subduction slip (m).  Defaults to 40.
    bounds : tuple or None, optional
        ``(x_min, y_min, x_max, y_max)`` map extent.
    fading_increment : float, optional
        Base of the exponential alpha decay.  Defaults to 2.0.
    time_to_threshold : float, optional
        Look-back window in years.  Defaults to 10.
    plot_log : bool, optional
        If ``True``, use a log colour scale.  Defaults to ``False``.
    log_min : float, optional
        Lower bound for the log scale.  Defaults to 1.
    log_max : float, optional
        Upper bound for the log scale.  Defaults to 100.
    min_slip_value : float or None, optional
        Minimum slip to plot.
    plot_zeros : bool, optional
        If ``True`` (default), plot zero-slip patches.
    extra_sub_list : list or None, optional
        Extra subduction patch numbers to highlight.
    min_mw : float or None, optional
        Minimum magnitude filter.
    decimals : int, optional
        Decimal places for the year label.  Defaults to 1.
    subplot_name : str, optional
        Key for the main axes in the ``axes`` dict.
        Defaults to ``"main_figure"``.

    Returns
    -------
    frame_num : int
        The input frame index.
    fig : matplotlib.figure.Figure or None
        Rendered figure, or ``None`` if no events fall in the window.
    """
    frame_time_seconds = frame_time * seconds_per_year

    shortened_cat = catalogue.filter_df(min_t0=frame_time_seconds - time_to_threshold * seconds_per_year,
                                        max_t0=frame_time_seconds,
                                        min_mw=min_mw).copy(deep=True)


    if shortened_cat.empty:
        return frame_num, None
    else:
        loaded_subplots = pickle.load(open(pickled_background, "rb"))
        print(frame_num)

        fig, axes = loaded_subplots
        slider_ax = axes["slider"]
        time_slider = Slider(
            slider_ax, 'Year', start_time - step_size, end_time + step_size, valinit=start_time - step_size,
            valstep=step_size)
        time_slider.valtext.set_visible(False)
        year_ax = axes["year"]
        year_text = year_ax.text(0.5, 0.5, str(int(0)), horizontalalignment='center', verticalalignment='center',
                                fontsize=12)
        if decimals == 0:
            year_text.set_text(str(int(round(frame_time, 0))))
        else:
            year_text.set_text(f"{frame_time:.{decimals}f}")
        time_slider.set_val(frame_time)

        shortened_cat["diff_t0"] = np.abs(shortened_cat["t0"] - frame_time_seconds)
        sorted_indices = shortened_cat.sort_values(by="diff_t0", ascending=False).index
        events_for_plot = catalogue.events_by_number(sorted_indices.tolist(), fault_model)
        for event in events_for_plot:

            alpha = calculate_alpha((frame_time - event.t0  / seconds_per_year), fading_increment)
            print(event.t0, alpha)
            event.plot_slip_2d(subplots=(fig, axes[subplot_name]), global_max_slip=global_max_slip,
                               global_max_sub_slip=global_max_sub_slip, bounds=bounds, plot_log_scale=plot_log,
                               log_min=log_min, log_max=log_max, min_slip_value=min_slip_value, plot_zeros=plot_zeros,
                               extra_sub_list=extra_sub_list, alpha=alpha)
            
        return frame_num, fig


def write_animation_frames(start_time, end_time, step_size, catalogue: RsqSimCatalogue, fault_model: RsqSimMultiFault,
                            pickled_background: str, subduction_cmap: str = "plasma", crustal_cmap: str = "viridis",
                            global_max_slip: int = 10, global_max_sub_slip: int = 40,
                            bounds: tuple = None, fading_increment: float = 2.0, time_to_threshold: float = 10.,
                            plot_log: bool = False, log_min: float = 1., log_max: float = 100.,
                            min_slip_value: float = None, plot_zeros: bool = False, extra_sub_list: list = None,
                            min_mw: float = None, decimals: int = 1, subplot_name: str = "main_figure",
                            num_threads_plot: int = 4, frame_dir: str = "frames",
                           ):
        """
        Write all animation frames to PNG files in parallel.

        Iterates over time steps from ``start_time`` to ``end_time`` in
        ``step_size`` increments and submits each frame to a
        ``ThreadPoolExecutor``.  Frames without events are written
        separately after the parallel pass.

        Parameters
        ----------
        start_time : float
            Animation start time in years.
        end_time : float
            Animation end time in years.
        step_size : int
            Year increment per frame.
        catalogue : RsqSimCatalogue
            Event catalogue.
        fault_model : RsqSimMultiFault
            Fault model for slip distributions.
        pickled_background : str
            Path to a pickled ``(fig, ax)`` background file.
        subduction_cmap : str, optional
            Colourmap for subduction slip.  Defaults to ``"plasma"``.
        crustal_cmap : str, optional
            Colourmap for crustal slip.  Defaults to ``"viridis"``.
        global_max_slip : int, optional
            Maximum crustal slip (m).  Defaults to 10.
        global_max_sub_slip : int, optional
            Maximum subduction slip (m).  Defaults to 40.
        bounds : tuple or None, optional
            ``(x_min, y_min, x_max, y_max)`` map extent.
        fading_increment : float, optional
            Base of the exponential alpha decay.  Defaults to 2.0.
        time_to_threshold : float, optional
            Look-back window in years.  Defaults to 10.
        plot_log : bool, optional
            If ``True``, use a log colour scale.  Defaults to ``False``.
        log_min : float, optional
            Lower bound for the log scale.  Defaults to 1.
        log_max : float, optional
            Upper bound for the log scale.  Defaults to 100.
        min_slip_value : float or None, optional
            Minimum slip to plot.
        plot_zeros : bool, optional
            If ``False`` (default), skip zero-slip patches.
        extra_sub_list : list or None, optional
            Extra subduction patch numbers to highlight.
        min_mw : float or None, optional
            Minimum magnitude filter.
        decimals : int, optional
            Decimal places for the year label.  Defaults to 1.
        subplot_name : str, optional
            Key for the main axes dict.  Defaults to ``"main_figure"``.
        num_threads_plot : int, optional
            Thread count for parallel rendering.  Defaults to 4.
        frame_dir : str, optional
            Directory for output PNG frames.  Defaults to ``"frames"``.
        """
        steps = np.arange(start_time, end_time + step_size, step_size)
        frames = np.arange(len(steps))
        pool_kwargs = { "catalogue": catalogue, "fault_model": fault_model,
                       "pickled_background": pickled_background, "subduction_cmap": subduction_cmap,
                       "crustal_cmap": crustal_cmap, "global_max_slip": global_max_slip,
                       "global_max_sub_slip": global_max_sub_slip, "bounds": bounds,
                       "fading_increment": fading_increment, "time_to_threshold": time_to_threshold,
                       "plot_log": plot_log, "log_min": log_min, "log_max": log_max,
                       "min_slip_value": min_slip_value, "plot_zeros": plot_zeros,
                       "extra_sub_list": extra_sub_list, "min_mw": min_mw, "decimals": decimals,
                       "subplot_name": subplot_name}
        
        no_earthquakes = []
        frame_time_dict = {frame_i: frame_time for frame_i, frame_time in enumerate(steps)}
        frame_block_size = 500
        block_starts = np.arange(0, len(steps), frame_block_size)

        def handle_output(future):
            frame_i, fig_i = future.result()
                    
            if fig_i is not None:
                fig_i.savefig(f"{frame_dir}/frame{frame_i:04d}.png", format="png", dpi=100)
                plt.close(fig_i)
                print(f"Writing {frame_i}")
                
            else:
                no_earthquakes.append(frame_i)

        for start, end in zip(block_starts, block_starts + frame_block_size):
            with ThreadPoolExecutor(max_workers=num_threads_plot) as plot_executor:
                for frame_i, frame_time in zip(frames[start:end], steps[start:end]):
                    if not os.path.exists(f"{frame_dir}/frame{frame_i:04d}.png"):
                         submitted = plot_executor.submit(write_animation_frame, frame_i, frame_time, start_time, end_time, step_size, **pool_kwargs)
                         submitted.add_done_callback(handle_output)
                


        for frame_num in no_earthquakes:
            loaded_subplots = pickle.load(open(pickled_background, "rb"))
            print(frame_num)
            frame_time = frame_time_dict[frame_num]

            fig, axes = loaded_subplots
            slider_ax = axes["slider"]
            time_slider = Slider(
                slider_ax, 'Year', start_time - step_size, end_time + step_size, valinit=start_time - step_size,
                valstep=step_size)
            time_slider.valtext.set_visible(False)
            year_ax = axes["year"]
            year_text = year_ax.text(0.5, 0.5, str(int(0)), horizontalalignment='center', verticalalignment='center',
                                    fontsize=12)
            if decimals == 0:
                year_text.set_text(str(int(round(frame_time, 0))))
            else:
                year_text.set_text(f"{frame_time:.{decimals}f}")
            time_slider.set_val(frame_time)
            fig.savefig(f"{frame_dir}/frame{frame_num:04d}.png", dpi=100)
            plt.close(fig)
        



def calculate_alpha(time_since_new, fading_increment):
    """
    Compute the opacity for an event that occurred ``time_since_new`` steps ago.

    Parameters
    ----------
    time_since_new : float
        Number of time steps since the event occurred.
    fading_increment : float
        Base of the exponential decay; higher values fade faster.

    Returns
    -------
    float
        Alpha value clamped to ``[0, 1]``.
    """
    alpha = 1 / (fading_increment ** time_since_new)
    if alpha > 1:
        alpha = 1.
    return alpha


def calculate_fading_increment(time_to_threshold, threshold):
    """
    Compute the fading increment so that alpha reaches ``threshold`` after ``time_to_threshold`` steps.

    Parameters
    ----------
    time_to_threshold : float
        Number of time steps until the event fades to ``threshold``.
    threshold : float
        Target alpha value after ``time_to_threshold`` steps
        (e.g. 0.01 for near-invisible).

    Returns
    -------
    float
        The fading increment base to pass to :func:`calculate_alpha`.
    """
    return (1 / threshold) ** (1 / time_to_threshold)

