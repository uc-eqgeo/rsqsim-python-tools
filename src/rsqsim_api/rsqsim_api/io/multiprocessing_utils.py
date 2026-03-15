"""Worker functions for constructing fault patches in parallel multiprocessing pools."""
import numpy as np
from rsqsim_api.fault.patch import RsqSimTriangularPatch


def array_to_patch(arg_ls: list):
    """
    Construct an RsqSimTriangularPatch from a flat argument list.

    Intended for use as a worker function in a multiprocessing pool,
    where arguments must be passed as a single iterable.

    Parameters
    ----------
    arg_ls :
        List of five elements: ``[patch_num, triangle, fault,
        strike_slip, dip_slip]``.

        - *patch_num* : int — patch index.
        - *triangle* : array-like of length 9 — flattened (3×3) vertex array.
        - *fault* : fault segment owning this patch.
        - *strike_slip* : float — strike-slip component in metres.
        - *dip_slip* : float — dip-slip component in metres.

    Returns
    -------
    patch_num : int
        The patch index (passed through for result ordering).
    patch : RsqSimTriangularPatch
        Constructed patch object.
    """
    patch_num, triangle, fault, strike_slip, dip_slip = arg_ls
    triangle3 = triangle.reshape(3, 3)
    patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num, strike_slip=strike_slip,
                                  dip_slip=dip_slip)
    return patch_num, patch


def array_to_patch_rake(arg_ls: list):
    """
    Construct an RsqSimTriangularPatch from a rake-based argument list.

    Decomposes a total slip magnitude and rake angle into strike-slip
    and dip-slip components before constructing the patch.  Intended for
    use as a worker function in a multiprocessing pool.

    Parameters
    ----------
    arg_ls :
        List of five elements: ``[patch_num, triangle, fault, rake,
        normalize_slip]``.

        - *patch_num* : int — patch index.
        - *triangle* : array-like of length 9 — flattened (3×3) vertex array.
        - *fault* : fault segment owning this patch.
        - *rake* : float — rake angle in degrees.
        - *normalize_slip* : float — total slip magnitude in metres.

    Returns
    -------
    patch_num : int
        The patch index (passed through for result ordering).
    patch : RsqSimTriangularPatch
        Constructed patch object with strike-slip and dip-slip set.
    """
    patch_num, triangle, fault, rake, normalize_slip = arg_ls
    triangle3 = triangle.reshape(3, 3)
    strike_slip = np.cos(np.radians(rake)) * normalize_slip
    dip_slip = np.sin(np.radians(rake)) * normalize_slip
    patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num, strike_slip=strike_slip,
                                  dip_slip=dip_slip)
    return patch_num, patch
