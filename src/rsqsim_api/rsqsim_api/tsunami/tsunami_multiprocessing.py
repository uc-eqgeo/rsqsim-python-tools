from rsqsim_api.containers.fault import RsqSimMultiFault, RsqSimSegment
import multiprocessing as mp
from typing import Union
import h5py
import numpy as np

sentinel = None


def multiprocess_gf_to_hdf(fault: Union[RsqSimSegment, RsqSimMultiFault], x_sites: np.ndarray, y_sites: np.ndarray,
                           out_file: str, z_sites: np.ndarray = None, slip_magnitude: Union[float, int] = 1.):
    # Check sites arrays
    assert all([isinstance(a, np.ndarray) for a in [x_sites, y_sites]])
    assert x_sites.shape == y_sites.shape
    assert x_sites.ndim <= 2

    if z_sites is not None:
        assert isinstance(z_sites, np.ndarray)
        assert z_sites.shape == x_sites.shape
    else:
        z_sites = np.zeros(x_sites.shape)

    n_patches = len(fault.patch_dic)

    if x_sites.ndim == 2:
        x_array = x_sites.flatten()
        y_array = y_sites.flatten()
        z_array = z_sites.flatten()
        dset_shape = (n_patches, x_sites.shape[0], x_sites.shape[1])
    else:
        x_array = x_sites
        y_array = y_sites
        z_array = z_sites
        dset_shape = (n_patches, x_sites.size)

    num_processes = int(np.round(mp.cpu_count() / 2))
    jobs = []
    out_queue = mp.Queue()
    in_queue = mp.Queue()
    output_proc = mp.Process(target=handle_output, args=(out_queue, out_file, dset_shape))
    output_proc.start()

    for i in range(num_processes):
        p = mp.Process(target=patch_greens_functions,
                       args=(in_queue, x_array, y_array, z_array, out_queue, dset_shape, slip_magnitude))
        jobs.append(p)
        p.start()

    for patch_i, patch in enumerate(fault.patch_outlines):
        in_queue.put((patch_i, patch))

    for i in range(num_processes):
        in_queue.put(sentinel)

    for p in jobs:
        p.join()

    out_queue.put(None)

    output_proc.join()


def handle_output(output_queue: mp.Queue, output_file: str, dset_shape: tuple):
    f = h5py.File(output_file, "w")
    ds_dset = f.create_dataset("dip_slip", shape=dset_shape, dtype="f")
    ss_dset = f.create_dataset("strike_slip", shape=dset_shape, dtype="f")

    while True:
        args = output_queue.get()
        if args:
            index, dip_slip, strike_slip = args
            ds_dset[index] = dip_slip
            ss_dset[index] = strike_slip
        else:
            break
    f.close()


def patch_greens_functions(in_queue: mp.Queue, x_sites: np.ndarray, y_sites: np.ndarray,
                           z_sites: np.ndarray,
                           out_queue: mp.Queue, grid_shape: tuple, slip_magnitude: Union[int, float] = 1):
    while True:
        queue_contents = in_queue.get()
        if queue_contents:
            index, patch = queue_contents
            print(patch.patch_number)
            ds_array, ss_array = patch.calculate_tsunami_greens_functions(x_sites, y_sites, z_sites,
                                                                          slip_magnitude=slip_magnitude)
            ds_grid = ds_array.reshape(grid_shape[1:])
            ss_grid = ss_array.reshape(grid_shape[1:])

            out_queue.put((index, ds_grid, ss_grid))
        else:
            break
