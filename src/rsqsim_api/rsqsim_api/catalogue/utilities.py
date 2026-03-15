"""
Utility functions for RSQSim catalogue statistics and seismological calculations.

Provides magnitude–moment conversions, b-value estimation, stress drop
and scaling-parameter calculations, and helper functions for computing
weighted circular statistics.
"""
import numpy as np
import pandas as pd
from numba import njit, int32

def calculate_b_value_kijko_smit(magnitudes: np.ndarray, min_mw: float = 0.0, max_mw: float = 10.0):
    """
    Estimate the Gutenberg-Richter b-value using the Kijko and Smit (2012) method.

    Parameters
    ----------
    magnitudes : numpy.ndarray
        Array of moment magnitudes.
    min_mw : float, optional
        Lower magnitude cutoff.  Events below this are excluded.
        Defaults to 0.0.
    max_mw : float, optional
        Upper magnitude cutoff.  Events above this are excluded.
        Defaults to 10.0.

    Returns
    -------
    float
        Estimated b-value (log₁₀ base).
    """
    magnitudes = np.array(magnitudes)
    trimmed_magnitudes = magnitudes[(magnitudes >= min_mw) & (magnitudes <= max_mw)]
    mw = trimmed_magnitudes
    mu1 = mw.mean()
    mu2 = np.sum((mw - mu1) ** 2) / mw.size
    mu3 = np.sum((mw - mu1) ** 3) / mw.size
    beta = 2 * mu2 / mu3
    b = beta / np.log(10)
    return b

def calculate_scaling_c(magnitudes: np.ndarray, areas: np.ndarray):
    """
    Calculate the Gutenberg-Richter scaling parameter c.

    The parameter ``c`` is defined as ``Mw - log10(area) + 6``, where
    ``area`` is in m².

    Parameters
    ----------
    magnitudes : numpy.ndarray
        Array of moment magnitudes.
    areas : numpy.ndarray
        Array of rupture areas in m².

    Returns
    -------
    numpy.ndarray
        Scaling parameter c for each event.
    """
    magnitudes = np.array(magnitudes)
    areas = np.array(areas)
    return magnitudes - np.log10(areas) + 6.

def calculate_stress_drop(seismic_moments: np.ndarray, areas: np.ndarray, stress_c=2.44):
    """
    Calculate stress drop from seismic moments and rupture areas.

    Uses the relation ``Δσ = C * M0 / A^1.5``.  Typical values of ``C``
    are 2.44 for a circular crack, and 2.53, 3.02, 5.21 for rectangular
    cracks with aspect ratios of 1, 4, and 16, respectively.

    Parameters
    ----------
    seismic_moments : numpy.ndarray
        Scalar seismic moments in N·m.
    areas : numpy.ndarray
        Rupture areas in m².
    stress_c : float, optional
        Shape constant.  Defaults to 2.44 (circular crack).

    Returns
    -------
    numpy.ndarray
        Stress drop in Pa for each event.
    """
    seismic_moments = np.array(seismic_moments)
    areas = np.array(areas)
    return  stress_c * (seismic_moments / areas**1.5)

def summary_statistics(dataframe: pd.DataFrame, stress_c: float = 2.44):
    """
    Compute summary statistics for a catalogue DataFrame.

    Calculates the 5th, 50th, and 95th percentiles of the scaling
    parameter c and the stress drop, together with the maximum magnitude.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Catalogue DataFrame with columns ``"mw"``, ``"area"``
        (m²), and ``"m0"`` (N·m).
    stress_c : float, optional
        Shape constant passed to :func:`calculate_stress_drop`.
        Defaults to 2.44.

    Returns
    -------
    pandas.Series
        Series with index
        ``["Max Mw, 5th C, 50th C, 95th C, 5th SD, 50th SD, 95th SD"]``
        where ``C`` is the scaling parameter and ``SD`` is the stress
        drop in MPa.
    """
    magnitudes = np.array(dataframe["mw"])
    areas = np.array(dataframe["area"])
    seismic_moments = np.array(dataframe["m0"])
    c_value = calculate_scaling_c(magnitudes, areas)
    stress_drop = calculate_stress_drop(seismic_moments, areas, stress_c=stress_c)
    c_value_percentiles = np.percentile(c_value, [5, 50, 95])
    stress_drop_percentiles = np.percentile(stress_drop, [5, 50, 95]) / 1e6
    labels = ["Max Mw, 5th C, 50th C, 95th C, 5th SD, 50th SD, 95th SD"]
    return pd.Series([max(magnitudes), *c_value_percentiles, *stress_drop_percentiles], index=labels)

@njit
def mw_to_m0(magnitudes: np.ndarray):
    """
    Convert moment magnitude to scalar seismic moment.

    Uses the relation ``M0 = 10^(1.5 * Mw + 9.05)``.

    Parameters
    ----------
    magnitudes : numpy.ndarray
        Array of moment magnitudes.

    Returns
    -------
    numpy.ndarray
        Scalar seismic moments in N·m.
    """
    return 10 ** (1.5 * magnitudes + 9.05)

@njit
def m0_to_mw(seismic_moment: float):
    """
    Convert scalar seismic moment to moment magnitude.

    Uses the relation ``Mw = (log10(M0) - 9.05) / 1.5``.

    Parameters
    ----------
    seismic_moment : float
        Scalar seismic moment in N·m.

    Returns
    -------
    float
        Moment magnitude.
    """
    return (np.log10(seismic_moment) - 9.05) / 1.5

def weighted_circular_mean(azimuths: np.ndarray, weights: np.ndarray):
    """
    Compute the weighted circular (angular) mean of a set of azimuths.

    Parameters
    ----------
    azimuths : numpy.ndarray
        Azimuth values in degrees.
    weights : numpy.ndarray
        Weights for each azimuth (e.g. patch areas or seismic moments).

    Returns
    -------
    float
        Weighted circular mean azimuth in degrees.
    """
    mean_sin = np.average(np.sin(np.radians(azimuths)), weights=weights)
    mean_cos = np.average(np.cos(np.radians(azimuths)), weights=weights)
    mean_az = np.degrees(np.arctan2(mean_sin, mean_cos))
    return mean_az

@njit
def median_cumulant(m0_array: np.ndarray, mw_array: np.ndarray):
    """
    Compute the median cumulant magnitude for a set of events on a patch.

    Sorts events by M0, forms the normalised cumulative sum of Mw, and
    returns the Mw at the index where the cumulative sum first reaches
    0.5.

    Parameters
    ----------
    m0_array : numpy.ndarray
        Scalar seismic moments (N·m) for all events rupturing a patch.
    mw_array : numpy.ndarray
        Corresponding moment magnitudes.

    Returns
    -------
    float
        Moment magnitude at the median of the cumulative M0 distribution.
    """
    # Sort the arrays by M0
    sorted_m0_indices = np.argsort(m0_array)
    sorted_m0 = m0_array[sorted_m0_indices]
    sorted_mw = mw_array[sorted_m0_indices]
    # Calculate the cumulative sum moment of the sorted m0
    norm_cumulative_m0 = np.cumsum(sorted_mw) / sorted_m0[-1]
    # Find median cumulant (last index where norm_cumulative_m0 < 0.5)
    med_cumulant_index = np.flatnonzero(norm_cumulative_m0 < 0.5)[-1]
    # Return MW for relevant index
    return sorted_mw[med_cumulant_index]

@njit(int32[:](int32[:], int32[:]))
def jit_intersect(l1, l2):
    """
    JIT-compiled intersection of two integer arrays.

    Returns the unique values that appear in both ``l1`` and ``l2``.

    Parameters
    ----------
    l1 : numpy.ndarray of int32
        First integer array.
    l2 : numpy.ndarray of int32
        Second integer array.

    Returns
    -------
    numpy.ndarray of int32
        Sorted unique values present in both arrays.
    """
    l3 = np.array([i for i in l1 for j in l2 if i == j])
    return np.unique(l3) # and i not in crossSec]

def mw_from_area_and_scaling_c(area: float, c: float):
    """
    Calculate moment magnitude from rupture area and scaling parameter c.

    Uses the relation ``Mw = log10(area) - 6 + c``.

    Parameters
    ----------
    area : float
        Rupture area in m².
    c : float
        Scaling parameter (typically ~4.2 for New Zealand crustal faults).

    Returns
    -------
    float
        Moment magnitude.
    """
    return np.log10(area) - 6. + c


def slip_from_area_and_scaling_c(area: float, c: float, mu: float =3.e10):
    """
    Calculate mean slip from rupture area and scaling parameter c.

    Computes the seismic moment from the area and ``c``, then divides by
    ``mu * area`` to give the average slip.

    Parameters
    ----------
    area : float
        Rupture area in m².
    c : float
        Scaling parameter (typically ~4.2 for New Zealand crustal faults).
    mu : float, optional
        Shear modulus in Pa.  Defaults to 3×10¹⁰ Pa (30 GPa).

    Returns
    -------
    float
        Average slip in metres.
    """

    m0 = mw_to_m0(mw_from_area_and_scaling_c(area, c))
    return m0 / (mu * area)
