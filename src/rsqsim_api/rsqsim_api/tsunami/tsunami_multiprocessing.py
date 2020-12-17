from rsqsim_api.containers.fault import RsqSimMultiFault, RsqSimSegment
import multiprocessing as mp
from typing import Union
import h5py
import netCDF4 as nc
import numpy as np
import random
sentinel = None


def multiprocess_gf_to_hdf(fault: Union[RsqSimSegment, RsqSimMultiFault], x_range: np.ndarray, y_range: np.ndarray,
                           out_file_prefix: str, x_grid: np.ndarray = None, y_grid: np.ndarray = None, z_grid: np.ndarray = None, slip_magnitude: Union[float, int] = 1.,
                           num_processors: int = None, num_write: int = 8):
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
    assert len(dset_shape) == 3
    assert len(patch_indices) == dset_shape[0]

    dset = nc.Dataset(output_file, "w")
    for dim, dim_len in zip(("npatch", "y", "x"), dset_shape):
        dset.createDimension(dim, dim_len)
    patch_var = dset.createVariable("index", np.int, ("npatch",))
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
                           out_queue_dic: dict, grid_shape: tuple, slip_magnitude: Union[int, float] = 1):
    while True:
        queue_contents = in_queue.get()
        if queue_contents:
            file_no, file_index, patch_number, patch = queue_contents

            out_queue_dic[file_no].put((file_index, patch_number,
                                        patch.calculate_tsunami_greens_functions(x_sites, y_sites, z_sites, grid_shape,
                                                                                 slip_magnitude=slip_magnitude)))
        else:
            break
