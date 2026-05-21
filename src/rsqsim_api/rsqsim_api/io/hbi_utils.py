"""
Readers for HBI (Hierarchical-matrix Boundary Integral) earthquake-sequence
simulator output, exposing it through the same fault/catalogue interfaces
the rest of ``rsqsim_api`` uses.

HBI is documented at https://github.com/sozawa94/hbi. For the
``3dnt``/``3dht`` problem types the fault geometry is supplied as an ASCII
STL file, and per-element data are written to a set of binary/ASCII
stream files keyed by an integer ``filenumber``:

================  ========================================================
File              Contents
================  ========================================================
``xyzN.dat``      Element-centre coordinates (km), reordered by the
                  H-matrix.  ``N`` rows of ``x y z``.
``eventN.dat``    Earthquake catalogue: ``event_num time_step onset_time
                  Mw hypocenter_id``.  ``hypocenter_id`` is a 1-based row
                  index into ``xyzN.dat``.
``EQslipN.dat``   Per-event slip per element, ``float64`` stream of
                  ``n_events * n_elements`` values, event-major.
``timeN.dat``     Field-snapshot timing: ``time_step time(s)``.
================  ========================================================

Coordinate units inside HBI are kilometres; this module converts to metres
on read so the rest of ``rsqsim_api`` (which uses metres internally) can
consume the data without further conversion.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from rsqsim_api.fault.multifault import RsqSimMultiFault
from rsqsim_api.fault.segment import RsqSimSegment
from rsqsim_api.io.read_utils import read_stl


HBI_EVENT_COLUMNS = ["event_num", "time_step", "onset_time", "mw", "hypocenter_id"]


def read_hbi_event_file(event_file: str) -> pd.DataFrame:
    """
    Parse an HBI ``eventN.dat`` file into a DataFrame.

    Parameters
    ----------
    event_file : str
        Path to the ``eventN.dat`` file.

    Returns
    -------
    pandas.DataFrame
        Columns ``event_num`` (int), ``time_step`` (int),
        ``onset_time`` (s, float), ``mw`` (float),
        ``hypocenter_id`` (1-based int).
    """
    assert os.path.exists(event_file), f"Event file not found: {event_file}"
    arr = np.loadtxt(event_file, ndmin=2)
    df = pd.DataFrame({
        "event_num": arr[:, 0].astype(np.int64),
        "time_step": arr[:, 1].astype(np.int64),
        "onset_time": arr[:, 2].astype(np.float64),
        "mw": arr[:, 3].astype(np.float64),
        "hypocenter_id": arr[:, 4].astype(np.int64),
    })
    return df


def read_hbi_xyz_file(xyz_file: str, to_metres: bool = True) -> np.ndarray:
    """
    Parse an HBI ``xyzN.dat`` element-centre file.

    Parameters
    ----------
    xyz_file : str
        Path to the ``xyzN.dat`` file.
    to_metres : bool, optional
        If ``True`` (default), multiply by 1000 to convert from HBI's
        native kilometres to metres.

    Returns
    -------
    numpy.ndarray of shape (n_elements, 3)
        Element-centre coordinates in metres (or km if ``to_metres`` is
        ``False``), in the H-matrix ordering used by all other HBI
        per-element output files.
    """
    assert os.path.exists(xyz_file), f"xyz file not found: {xyz_file}"
    xyz = np.loadtxt(xyz_file, ndmin=2)
    assert xyz.shape[1] == 3, f"Expected 3 columns in {xyz_file}, got {xyz.shape[1]}"
    if to_metres:
        xyz = xyz * 1000.0
    return xyz


def read_hbi_eqslip_file(eqslip_file: str, n_events: int, n_elements: int,
                         dtype: str = "<f8") -> np.ndarray:
    """
    Read a binary ``EQslipN.dat`` slip-per-event-per-element stream.

    HBI writes ``EQslipN.dat`` as a flat Fortran stream of length
    ``n_events * n_elements`` with the per-event slip vector contiguous
    (event-major), so reshaping as ``(n_events, n_elements)`` in C order
    yields one event per row.

    Parameters
    ----------
    eqslip_file : str
        Path to the ``EQslipN.dat`` file.
    n_events : int
        Number of events (rows in ``eventN.dat``).
    n_elements : int
        Number of mesh elements (rows in ``xyzN.dat``).
    dtype : str, optional
        NumPy dtype string for the binary entries.  Defaults to
        little-endian float64 (``"<f8"``); use ``">f8"`` for big-endian.

    Returns
    -------
    numpy.ndarray of shape (n_events, n_elements)
        Per-event slip (m) for each element, in the H-matrix ordering of
        ``xyzN.dat``.
    """
    assert os.path.exists(eqslip_file), f"EQslip file not found: {eqslip_file}"
    expected_bytes = n_events * n_elements * np.dtype(dtype).itemsize
    actual_bytes = os.path.getsize(eqslip_file)
    assert actual_bytes == expected_bytes, (
        f"{eqslip_file} size {actual_bytes} bytes does not match "
        f"n_events={n_events} * n_elements={n_elements} * "
        f"{np.dtype(dtype).itemsize} = {expected_bytes} bytes"
    )
    flat = np.fromfile(eqslip_file, dtype=dtype)
    return flat.reshape((n_events, n_elements))


def _match_stl_triangles_to_xyz(triangle_centroids: np.ndarray,
                                xyz: np.ndarray,
                                tolerance: float = 10.0) -> np.ndarray:
    """
    Compute a permutation that reorders STL triangles to match xyz row order.

    For each row of ``xyz`` finds the nearest STL triangle centroid; the
    returned array ``perm`` has the property that
    ``triangle_centroids[perm[i]]`` is the centroid closest to
    ``xyz[i]``.  Asserts a one-to-one match and that all matches are
    within ``tolerance``.

    Parameters
    ----------
    triangle_centroids : numpy.ndarray of shape (n, 3)
        STL triangle centroids (same units as ``xyz``).
    xyz : numpy.ndarray of shape (n, 3)
        HBI ``xyzN.dat`` rows in H-matrix order.
    tolerance : float, optional
        Maximum allowed distance (same units as ``xyz``) for a centroid
        to count as matching an ``xyz`` row.  Defaults to 1 metre.
    """
    assert triangle_centroids.shape == xyz.shape, (
        f"STL has {len(triangle_centroids)} triangles but xyz has "
        f"{len(xyz)} rows; they must match one-to-one."
    )
    tree = KDTree(triangle_centroids)
    distances, perm = tree.query(xyz, k=1)
    if np.max(distances) > tolerance:
        worst = int(np.argmax(distances))
        raise ValueError(
            f"STL triangle centroid for xyz row {worst} is "
            f"{distances[worst]:.3f} away (tolerance {tolerance}). "
            "STL and xyz files may not describe the same mesh."
        )
    unique = np.unique(perm)
    if len(unique) != len(perm):
        raise ValueError(
            "STL triangles do not map one-to-one to xyz rows "
            f"({len(unique)} unique matches for {len(perm)} rows). "
            "Check the STL and xyz files describe the same mesh."
        )
    return perm


def multifault_from_hbi_stl(stl_file: str, xyz_file: str,
                            rake: float = 90.0,
                            fault_name: str = "hbi_fault",
                            min_point_sep: float = 1.0,
                            match_tolerance: float = 10.0,
                            crs: int = 2193) -> RsqSimMultiFault:
    """
    Build a single-segment :class:`~rsqsim_api.fault.multifault.RsqSimMultiFault`
    from an HBI STL geometry file, with patches indexed to match ``xyzN.dat``.

    Triangles are reordered so that ``multifault.patch_dic[i]`` corresponds
    to row ``i`` of ``xyzN.dat`` (and therefore column ``i`` of
    ``EQslipN.dat``).  Vertex coordinates are converted from HBI's
    kilometres to metres.

    Parameters
    ----------
    stl_file : str
        Path to the HBI STL geometry file (ASCII).  In HBI's
        ``3dnt``/``3dht`` problems this is the same STL passed to the
        simulator.
    xyz_file : str
        Path to the corresponding ``xyzN.dat`` file used to derive the
        H-matrix ordering.
    rake : float, optional
        Uniform rake angle (degrees) assigned to every patch.  Defaults
        to 90 (pure reverse).  HBI's ``crake`` convention is used: 90 =
        reverse, 0 = left-lateral, -180 = right-lateral, -90 = normal.
    fault_name : str, optional
        Name attached to the resulting segment.
    min_point_sep : float, optional
        Vertex-merge tolerance for :func:`read_stl`, in **metres**
        (vertices closer than this after the km→m conversion are
        treated as duplicates).  Defaults to 1 m.
    match_tolerance : float, optional
        Maximum allowed distance (metres) between an STL triangle
        centroid and its matching ``xyzN.dat`` row.  Defaults to 1 m.
    crs : int, optional
        EPSG code attached to the multifault.  Defaults to 2193 (NZTM).

    Returns
    -------
    RsqSimMultiFault
        Single-segment multifault with ``n_elements`` triangular
        patches ordered to match ``xyzN.dat``.
    """
    triangles_km = read_stl(stl_file, min_point_sep=min_point_sep / 1000.0)
    triangles_m = triangles_km * 1000.0
    xyz_m = read_hbi_xyz_file(xyz_file, to_metres=True)

    centroids_m = triangles_m.reshape(-1, 3, 3).mean(axis=1)
    perm = _match_stl_triangles_to_xyz(centroids_m, xyz_m,
                                       tolerance=match_tolerance)
    triangles_ordered = triangles_m[perm]

    segment = RsqSimSegment.from_triangles(
        triangles_ordered,
        segment_number=0,
        patch_numbers=np.arange(len(triangles_ordered)),
        fault_name=fault_name,
        rake=rake,
    )
    return RsqSimMultiFault([segment], crs=crs)


def hbi_events_to_catalogue_df(events: pd.DataFrame, xyz_m: np.ndarray,
                               eqslip: np.ndarray,
                               patch_areas_m2: np.ndarray) -> pd.DataFrame:
    """
    Build the standard RSQSim catalogue DataFrame from parsed HBI inputs.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`read_hbi_event_file`.
    xyz_m : numpy.ndarray of shape (n_elements, 3)
        Element-centre coordinates in metres (H-matrix order).
    eqslip : numpy.ndarray of shape (n_events, n_elements)
        Per-event slip (m).
    patch_areas_m2 : numpy.ndarray of shape (n_elements,)
        Area of each patch in m², in the same H-matrix order as
        ``xyz_m``.

    Returns
    -------
    pandas.DataFrame
        Columns ``t0, m0, mw, x, y, z, area, dt`` with one row per
        event.  ``m0`` is derived from ``mw`` via
        ``M0 = 10**(1.5*Mw + 9.05)``.  ``area`` is the summed area of
        all patches with non-zero slip; ``dt`` is set to ``0`` (HBI's
        ``eventN.dat`` does not record rupture duration).  Index is the
        0-based event ordinal (so ``event_num - 1``).
    """
    from rsqsim_api.catalogue.utilities import mw_to_m0

    hypo_idx0 = events["hypocenter_id"].to_numpy() - 1
    assert hypo_idx0.min() >= 0 and hypo_idx0.max() < len(xyz_m), (
        f"hypocenter_id out of range for xyz of length {len(xyz_m)}"
    )
    hypo_xyz = xyz_m[hypo_idx0]

    slip_mask = eqslip != 0
    area_per_event = slip_mask @ patch_areas_m2

    mw = events["mw"].to_numpy()
    df = pd.DataFrame({
        "t0": events["onset_time"].to_numpy(),
        "m0": mw_to_m0(mw),
        "mw": mw,
        "x": hypo_xyz[:, 0],
        "y": hypo_xyz[:, 1],
        "z": hypo_xyz[:, 2],
        "area": area_per_event,
        "dt": np.zeros(len(events), dtype=np.float64),
    })
    df.index = np.arange(len(events), dtype=np.int64)
    return df


def flatten_hbi_eqslip(eqslip: np.ndarray, onset_times: np.ndarray):
    """
    Flatten a dense per-event-per-patch slip matrix into RSQSim list arrays.

    Skips zero-slip entries so the resulting arrays look like RSQSim's
    native ``.pList``/``.eList``/``.dList``/``.tList`` representation,
    which only stores patches that actually slipped.

    All patches within a given event are assigned the event's onset
    time as their ``patch_time``; HBI's ``EQslipN.dat`` is per-event
    cumulative slip with no within-event rupture timing.

    Parameters
    ----------
    eqslip : numpy.ndarray of shape (n_events, n_elements)
        Per-event slip (m).  Rows are events in the order of
        ``eventN.dat`` (0-based event index).
    onset_times : numpy.ndarray of shape (n_events,)
        Event onset times (s) from ``eventN.dat``.

    Returns
    -------
    event_list : numpy.ndarray of int
        0-based event index repeated once per non-zero patch entry.
    patch_list : numpy.ndarray of int
        0-based patch index (column of ``eqslip``).
    patch_slip : numpy.ndarray of float64
        Slip (m) for each entry.
    patch_time_list : numpy.ndarray of float64
        Onset time (s) of the event each entry belongs to.
    """
    assert eqslip.ndim == 2
    assert onset_times.shape == (eqslip.shape[0],)

    rows, cols = np.nonzero(eqslip)
    event_list = rows.astype(np.int32)
    patch_list = cols.astype(np.int32)
    patch_slip = eqslip[rows, cols].astype(np.float64)
    patch_time_list = onset_times[rows].astype(np.float64)
    return event_list, patch_list, patch_slip, patch_time_list
