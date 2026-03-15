"""
Spline utilities for constructing idealised slip distributions on fault patches.

Provides functions that build :class:`scipy.interpolate.UnivariateSpline`
objects encoding along-fault and depth-dependent slip profiles, suitable
for imposing prescribed slip distributions in RSQSim or other fault models.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

def fault_edge_spline(max_slip: float, distance_width: float, total_width: float, min_slip_fraction: float = 0., line_stop_fraction: float = 0.5, gradient_change: float = 1.2,
                      spline_k: int = 3, spline_s: float = 0.0, resolution: float = 10.):
    """
    Build a univariate spline encoding slip as a function of along-fault distance.

    Constructs an interpolated spline that ramps from a minimum slip
    fraction at the fault edge up to the maximum slip, with a
    piecewise-linear interpolant shaped by the supplied parameters.

    Parameters
    ----------
    max_slip : float
        Maximum slip value (m or m/yr) at the centre of the fault.
    distance_width : float
        Along-fault distance (m) at which ``max_slip`` is reached.
    total_width : float
        Total along-fault width (m) of the spline domain.
    min_slip_fraction : float, optional
        Fraction of ``max_slip`` at the fault edge (distance = 0).
        Defaults to 0.
    line_stop_fraction : float, optional
        Fraction of ``distance_width`` at which the initial ramp
        levels off before the gradient change.  Defaults to 0.5.
    gradient_change : float, optional
        Multiplicative factor applied to the intermediate y value to
        create a smooth overshoot before the final ramp.  Defaults to
        1.2.
    spline_k : int, optional
        Degree of the smoothing spline.  Defaults to 3 (cubic).
    spline_s : float, optional
        Smoothing factor passed to
        :class:`~scipy.interpolate.UnivariateSpline`.  Defaults to
        0.0 (interpolating spline).
    resolution : float, optional
        Sample spacing (m) used when building the x-axis for
        interpolation.  Defaults to 10.

    Returns
    -------
    scipy.interpolate.UnivariateSpline
        Spline object; call with an x array (metres from edge) to
        evaluate the slip profile.
    """

    x_points = np.hstack([np.arange(0., distance_width * line_stop_fraction + resolution, resolution),
                            np.arange(distance_width, total_width + resolution, resolution)])

    x_for_interp = np.array([0., distance_width * line_stop_fraction, distance_width, total_width])
    y_for_interp = np.array([max_slip * min_slip_fraction, gradient_change *
                             (max_slip * min_slip_fraction + line_stop_fraction * max_slip * (1 - min_slip_fraction)),
                             max_slip, max_slip])

    interpolated_y = np.interp(x_points, x_for_interp, y_for_interp)
    out_spline = UnivariateSpline(x_points, interpolated_y, k=spline_k, s=spline_s)
    return out_spline

def fault_depth_spline(gradient_change_x: float, after_change_fract: float, resolution: float = 0.01,
                       after_change_gradient: float = 1.2, spline_k: int = 3, spline_s: float = 0.0):
    """
    Build a univariate spline encoding normalised slip as a function of normalised depth.

    Produces a spline on [0, 1] (normalised depth fraction) that is
    approximately flat at 1 down to ``gradient_change_x``, then ramps
    linearly to 0 at the base of the fault, with an optional gradient
    overshoot controlled by ``after_change_gradient``.

    Parameters
    ----------
    gradient_change_x : float
        Normalised depth (0–1) at which the gradient begins to change.
    after_change_fract : float
        Fraction of the remaining depth below ``gradient_change_x``
        over which the gradient change is applied.
    resolution : float, optional
        Sample spacing in normalised depth used when building the
        spline.  Defaults to 0.01.
    after_change_gradient : float, optional
        Multiplicative factor that controls the magnitude of the
        gradient overshoot.  Defaults to 1.2.
    spline_k : int, optional
        Degree of the smoothing spline.  Defaults to 3 (cubic).
    spline_s : float, optional
        Smoothing factor passed to
        :class:`~scipy.interpolate.UnivariateSpline`.  Defaults to
        0.0 (interpolating spline).

    Returns
    -------
    scipy.interpolate.UnivariateSpline
        Spline object; call with a normalised-depth array to evaluate
        the depth-dependent slip multiplier.
    """
    x_points = np.hstack([np.arange(0., gradient_change_x + resolution, resolution),
                            np.arange(1 - (1. - gradient_change_x) * after_change_fract, 1. + resolution, resolution)])
    x_for_interp = np.array([0., gradient_change_x, 1. - (1. - gradient_change_x) * after_change_fract, 1.])
    y_for_interp = np.array([1., 1., after_change_gradient * after_change_fract, 0.])
    interpolated_y = np.interp(x_points, x_for_interp, y_for_interp)
    out_spline = UnivariateSpline(x_points, interpolated_y, k=spline_k, s=spline_s)
    return out_spline
