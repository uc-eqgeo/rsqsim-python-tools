import numpy as np
from scipy.interpolate import UnivariateSpline

def fault_edge_spline(max_slip: float, distance_width: float, total_width: float, min_slip_fraction: float = 0., line_stop_fraction: float = 0.5, gradient_change: float = 1.2,
                      spline_k: int = 3, spline_s: float = 0.0, resolution: float = 10.):
    """
    Create a spline that can be used to create a slip distribution along the edge of a fault.
    @param max_slip:
    @param distance_width:
    @param min_slip_fraction:
    @param line_stop_fraction:
    @param gradient_change:
    @param spline_k:
    @param spline_s:
    @param resolution:
    @return:
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
    x_points = np.hstack([np.arange(0., gradient_change_x + resolution, resolution),
                            np.arange(1 - (1. - gradient_change_x) * after_change_fract, 1. + resolution, resolution)])
    x_for_interp = np.array([0., gradient_change_x, 1. - (1. - gradient_change_x) * after_change_fract, 1.])
    y_for_interp = np.array([1., 1., after_change_gradient * after_change_fract, 0.])
    interpolated_y = np.interp(x_points, x_for_interp, y_for_interp)
    out_spline = UnivariateSpline(x_points, interpolated_y, k=spline_k, s=spline_s)
    return out_spline