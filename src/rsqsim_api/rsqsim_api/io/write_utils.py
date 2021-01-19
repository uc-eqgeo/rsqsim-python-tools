import numpy as np
import pandas as pd
import os

import rsqsim_api.containers.fault


def write_catalogue_dataframe_and_arrays(prefix: str, catalogue, directory: str = None,
                                         write_index: bool = True):
    if directory is not None:
        assert os.path.exists(directory)
        dir_path = directory
    else:
        dir_path = ""

    assert isinstance(prefix, str)
    assert len(prefix) > 0
    if prefix[-1] != "_":
        prefix += "_"
    prefix_path = os.path.join(dir_path, prefix)
    df_file = prefix_path + "catalogue.csv"
    event_file = prefix_path + "events.npy"
    patch_file = prefix_path + "patches.npy"
    slip_file = prefix_path + "slip.npy"
    slip_time_file = prefix_path + "slip_time.npy"

    catalogue.catalogue_df.to_csv(df_file, index=write_index)
    for file, array in zip([event_file, patch_file, slip_file, slip_time_file],
                           [catalogue.event_list, catalogue.patch_list, catalogue.patch_slip,
                            catalogue.patch_time_list]):
        np.save(file, array)


def fit_plane_to_points(points: np.ndarray):
    """
    Finds best-fit plane through a set of points.
    A least-squares solution is used to find the solution to z = a*x + b*y + c.
    If the fault is nearly vertical, the solution is computed as x = a*y + b*z + c.
    The plane normal is then computed and the plane may be computed as:
    A*(x-x0) +B*(y-y0) + C*(z-z0) = 0, where (x0, y0, z0) is the plane_origin,
    and (A, B, C) is the plane normal.
    Returned values are:
                plane_normal:  Normal vector to plane (A, B, C)
                plane_origin:  Point on plane that may be considered as the plane origin
    """
    # Number of points and mean of points.
    num_points = points.shape[0]
    # points_mean = rsqsim_api.containers.fault.calculate_centre(points)
    points_mean = np.mean(points, axis=0)
    plane_origin = np.zeros(3, np.float64)
    inds = [0,1,2]
    c0 = np.array([0.0, 0.0, 5000.0], dtype=np.float64)
    c1 = np.array([5000.0, 0.0, 0.0], dtype=np.float64)
    plane_points = np.zeros((3,3), dtype=np.float64)

    # Form inversion arrays and compute least-squares solution.
    # First try assuming plane is not vertical.
    a = np.ones((num_points, 3), dtype=np.float64)
    a[:,0] = points[:,0]
    a[:,1] = points[:,1]
    b = points[:,2]
    (sol, resid, rank, single_vals) = np.linalg.lstsq(a, b, rcond=None)

    # If not full rank, fault is probably nearly vertical.
    if (rank < 3):
        a = np.ones((num_points, 3), dtype=np.float64)
        a[:,0] = points[:,1]
        a[:,1] = points[:,2]
        b = points[:,0]
        (sol, resid, rank, single_vals) = np.linalg.lstsq(a, b, rcond=None)
        inds = [1,2,0]

    # Compute plane origin using solution.
    plane_origin[inds[0]] = points_mean[inds[0]]
    plane_origin[inds[1]] = points_mean[inds[1]]
    plane_origin[inds[2]] = sol[0]*plane_origin[inds[0]] + sol[1]*plane_origin[inds[1]] + sol[2]

    # Compute coordinates corresponding to c0 and c1.
    c2 = sol[0]*c0 + sol[1]*c1 + sol[2]
    plane_points[:,inds[0]] = c0
    plane_points[:,inds[1]] = c1
    plane_points[:,inds[2]] = c2
    
    # Compute normal vector and normalize it.
    v1 = plane_points[1,:] - plane_points[0,:]
    v2 = plane_points[2,:] - plane_points[0,:]
    plane_normal = rsqsim_api.containers.fault.cross_3d(v1, v2)
    plane_normal /= rsqsim_api.containers.fault.norm_3d(plane_normal)

    return (plane_normal, plane_origin)
    
    
    
    
