"""
Multiprocessing utilities for computing tsunami Green's functions in parallel.

Distributes the per-patch Green's-function computation across multiple
worker processes and writes the results incrementally to a set of
netCDF output files via a producer/consumer pattern.
"""
from rsqsim_api.fault.multifault import RsqSimMultiFault, RsqSimSegment
import multiprocessing as mp
import h5py
import netCDF4 as nc
import numpy as np
import random
sentinel = None


def multiprocess_gf_to_hdf(fault: RsqSimSegment | RsqSimMultiFault, x_range: np.ndarray, y_range: np.ndarray,
                           out_file_prefix: str, x_grid: np.ndarray = None, y_grid: np.ndarray = None, z_grid: np.ndarray = None, slip_magnitude: float | int = 1.,
                           num_processors: int = None, num_write: int = 8):
    """
    Compute tsunami Green's functions for all patches and write to netCDF files.

    Distributes patch computations across ``num_processors`` worker
    processes and writes results to ``num_write`` netCDF output files
    via per-file output queues.  Patches are randomly shuffled before
    distribution to balance load.

    Parameters
    ----------
    fault : RsqSimSegment or RsqSimMultiFault
        Fault model containing the patches to process.
    x_range : numpy.ndarray of shape (nx,)
        1-D easting coordinate array (NZTM metres).
    y_range : numpy.ndarray of shape (ny,)
        1-D northing coordinate array (NZTM metres).
    out_file_prefix : str
        Prefix for output netCDF files; files are named
        ``{out_file_prefix}{i}.nc`` for ``i`` in
        ``range(num_write)``.
    x_grid : numpy.ndarray or None, optional
        2-D easting grid of shape ``(ny, nx)``.  If ``None``,
        constructed from ``x_range`` and ``y_range`` via meshgrid.
    y_grid : numpy.ndarray or None, optional
        2-D northing grid; must match ``x_grid`` shape.
    z_grid : numpy.ndarray or None, optional
        2-D elevation grid (m); defaults to all zeros.
    slip_magnitude : float or int, optional
        Unit slip magnitude used for the Green's function calculation.
        Defaults to 1.
    num_processors : int or None, optional
        Number of worker processes.  Defaults to half the available
        CPU count.
    num_write : int, optional
        Number of output netCDF files (and output processes).
        Defaults to 8.
    """
    assert all([isinstance(a, np.ndarray) for a in [x_range, y_range]])
    assert all([x_range.ndim == 1, y_range.ndim == 1])

    # Check sites arrays

    if all([a is not None for a in (x_grid, y_grid)]):
        assert all([isinstance(a, np.ndarray) for a in [x_grid, y_grid]])
        assert x_grid.shape == (y_range.size, x_range.size)
        assert x_grid.shape == y_grid.shape
        assert x_grid.ndim <= 2
    else:
        x_grid, y_grid = np.meshgrid(x_range, y_range)

    if z_grid is not None:
        assert isinstance(z_grid, np.ndarray)
        assert z_grid.shape == x_grid.shape
    else:
        z_grid = np.zeros(x_grid.shape)

    n_patches = len(fault.patch_dic)

    if x_grid.ndim == 2:
        x_array = x_grid.flatten()
        y_array = y_grid.flatten()
        z_array = z_grid.flatten()
        dset_shape = (n_patches, x_grid.shape[0], x_grid.shape[1])
    else:
        x_array = x_grid
        y_array = y_grid
        z_array = z_grid
        dset_shape = (n_patches, x_grid.size)

    if num_processors is None:
        num_processes = int(np.round(mp.cpu_count() / 2))
    else:
        assert isinstance(num_processors, int)
        num_processes = num_processors

    all_patch_ls = []
    if isinstance(fault, RsqSimSegment):
        for patch in fault.patch_outlines:
            all_patch_ls.append([patch.patch_number, patch])
    else:
        for patch_i, patch in fault.patch_dic.items():
            all_patch_ls.append([patch_i, patch])

    num_per_write = int(np.round(len(all_patch_ls) / num_write))
    all_patches_with_write_indices = []
    separate_write_index_dic = {}
    for i in range(num_write):
        range_min = i * num_per_write
        range_max = (i + 1) * num_per_write
        index_ls = []
        for file_index, patch_tuple in enumerate(all_patch_ls[range_min:range_max]):
            new_ls = [i, file_index] + patch_tuple
            all_patches_with_write_indices.append(new_ls)
            index_ls.append(patch_tuple[0])
        separate_write_index_dic[i] = np.array(index_ls)

    random.shuffle(all_patches_with_write_indices)

    out_queue_dic = {}
    out_proc_ls = []
    for i in range(num_write):

        patch_indices = separate_write_index_dic[i]
        dset_shape_i = (len(patch_indices), dset_shape[1], dset_shape[-1])
        out_queue = mp.Queue(maxsize=1000)
        out_file_name = out_file_prefix + "{:d}.nc".format(i)
        out_queue_dic[i] = out_queue
        output_proc = mp.Process(target=handle_output_netcdf, args=(out_queue, separate_write_index_dic[i],
                                                                    out_file_name, dset_shape_i, x_range, y_range))
        out_proc_ls.append(output_proc)
        output_proc.start()

    jobs = []
    in_queue = mp.Queue()
    for i in range(num_processes):
        p = mp.Process(target=patch_greens_functions,
                       args=(in_queue, x_array, y_array, z_array, out_queue_dic, dset_shape, slip_magnitude))
        jobs.append(p)
        p.start()

    for row in all_patches_with_write_indices:
        file_no, file_index, patch_index, patch = row
        in_queue.put((file_no, file_index, patch_index, patch))

    for i in range(num_processes):
        in_queue.put(sentinel)

    for p in jobs:
        p.join()

    for i in range(num_write):
        out_queue_dic[i].put(sentinel)
        out_proc_ls[i].join()

    in_queue.close()
    for i in range(num_write):
        out_queue_dic[i].close()


def handle_output(output_queue: mp.Queue, output_file: str, dset_shape: tuple):
    """
    Consumer process that writes sea-surface displacement data to an HDF5 file.

    Reads ``(index, vert_disp)`` tuples from the queue until the
    sentinel value is received.

    Parameters
    ----------
    output_queue : multiprocessing.Queue
        Queue delivering ``(index, disp_array)`` tuples.
    output_file : str
        Output HDF5 file path.
    dset_shape : tuple
        Shape of the ``"ssd_1m"`` dataset.
    """
    f = h5py.File(output_file, "w")
    disp_dset = f.create_dataset("ssd_1m", shape=dset_shape, dtype="f")

    while True:
        args = output_queue.get()
        if args:
            index, vert_disp = args
            disp_dset[index] = vert_disp
        else:
            break
    f.close()


def handle_output_netcdf(output_queue: mp.Queue, patch_indices: np.ndarray, output_file: str, dset_shape: tuple,
                         x_range: np.ndarray, y_range: np.ndarray):
    """
    Consumer process that writes sea-surface displacement data to a netCDF4 file.

    Creates a netCDF4 file with dimensions ``(npatch, y, x)`` and
    reads ``(index, patch_index, disp_array)`` tuples from the queue
    until the sentinel value is received.

    Parameters
    ----------
    output_queue : multiprocessing.Queue
        Queue delivering ``(local_index, patch_index, disp_array)``
        tuples.
    patch_indices : numpy.ndarray
        Array of global patch indices stored in this file.
    output_file : str
        Output netCDF4 file path.
    dset_shape : tuple of int
        Shape ``(n_patches, ny, nx)`` of the SSD variable.
    x_range : numpy.ndarray
        1-D easting coordinate array.
    y_range : numpy.ndarray
        1-D northing coordinate array.
    """
    assert len(dset_shape) == 3
    assert len(patch_indices) == dset_shape[0]

    dset = nc.Dataset(output_file, "w")
    dset.set_always_mask(False)
    for dim, dim_len in zip(("npatch", "y", "x"), dset_shape):
        dset.createDimension(dim, dim_len)
    patch_var = dset.createVariable("index", int, ("npatch",))
    dset.createVariable("x", np.float32, ("x",))
    dset.createVariable("y", np.float32, ("y",))
    dset["x"][:] = x_range
    dset["y"][:] = y_range

    patch_var[:] = patch_indices
    ssd = dset.createVariable("ssd", np.float32, ("npatch", "y", "x"), least_significant_digit=4)
    counter = 0
    num_patch = len(patch_indices)
    while True:
        args = output_queue.get()
        if args:
            index, patch_index, vert_disp = args
            assert patch_index in patch_indices
            ssd[index] = vert_disp
            counter += 1
            print("{:d}/{:d} complete".format(counter, num_patch))
        else:
            break
    dset.close()


def patch_greens_functions(in_queue: mp.Queue, x_sites: np.ndarray, y_sites: np.ndarray,
                           z_sites: np.ndarray,
                           out_queue_dic: dict, grid_shape: tuple, slip_magnitude: int | float = 1):
    """
    Worker process that computes Green's functions for patches received from the input queue.

    Reads ``(file_no, file_index, patch_number, patch)`` tuples from
    ``in_queue``, calls
    :meth:`~rsqsim_api.fault.patch.RsqSimTriangularPatch.calculate_tsunami_greens_functions`,
    and forwards the result to the appropriate output queue.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue of ``(file_no, file_index, patch_number, patch)``
        tuples.  A ``None`` sentinel signals termination.
    x_sites : numpy.ndarray
        Flattened easting coordinates of the output grid.
    y_sites : numpy.ndarray
        Flattened northing coordinates.
    z_sites : numpy.ndarray
        Flattened elevation coordinates.
    out_queue_dic : dict
        Mapping of file index to output queue.
    grid_shape : tuple
        Shape of the output displacement grid.
    slip_magnitude : int or float, optional
        Unit slip magnitude.  Defaults to 1.
    """
    while True:
        queue_contents = in_queue.get()
        if queue_contents:
            file_no, file_index, patch_number, patch = queue_contents

            out_queue_dic[file_no].put((file_index, patch_number,
                                        patch.calculate_tsunami_greens_functions(x_sites, y_sites, z_sites, grid_shape,
                                                                                 )))
        else:
            break
