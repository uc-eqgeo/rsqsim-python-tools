import numpy as np
import pandas as pd
from numba import njit, int32

def calculate_b_value_kijko_smit(magnitudes: np.ndarray, min_mw: float = 0.0, max_mw: float = 10.0):
    """
    Calculate b-value from magnitudes using the Kijko and Smit (2012) method.
    :param magnitudes:
    :param min_mw:
    :param max_mw:
    :return:
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
    Calculate the scaling parameter c for the Gutenberg-Richter relation.
    :param magnitudes:
    :param areas:
    :return:
    """
    magnitudes = np.array(magnitudes)
    areas = np.array(areas)
    return magnitudes - np.log10(areas) + 6.

def calculate_stress_drop(seismic_moments: np.ndarray, areas: np.ndarray, stress_c=2.44):
    """
    Calculate the stress drop from magnitudes and areas.
    C = 2.44 for a circular ruptured domain (or crack)
    C = 2.53, 3.02 and 5.21 for rectangular cracks with aspect ratios α = 1, 4 and 16, respectively
    :param seismic_moments:
    :param areas:
    :param mu:
    :param stress_c:
    :return:
    """
    seismic_moments = np.array(seismic_moments)
    areas = np.array(areas)
    return  stress_c * (seismic_moments / areas**1.5)

def summary_statistics(dataframe: pd.DataFrame, stress_c: float = 2.44):
    """
    Calculate summary statistics for a dataframe of events.
    :param dataframe:
    :param min_mw:
    :param max_mw:
    :param stress_c:
    :return:
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
    Convert magnitudes to seismic moment.
    :param magnitudes:
    :return:
    """
    return 10 ** (1.5 * magnitudes + 9.05)

@njit
def m0_to_mw(seismic_moment: float):
    """
    Convert seismic moment to magnitudes.
    :param seismic_moments:
    :return:
    """
    return (np.log10(seismic_moment) - 9.05) / 1.5

def weighted_circular_mean(azimuths: np.ndarray, weights: np.ndarray):
    mean_sin = np.average(np.sin(np.radians(azimuths)), weights=weights)
    mean_cos = np.average(np.cos(np.radians(azimuths)), weights=weights)
    mean_az = np.degrees(np.arctan2(mean_sin, mean_cos))
    return mean_az

@njit
def median_cumulant(m0_array: np.ndarray, mw_array: np.ndarray):
    """
    Calculate the median cumulant for a given array of moment magnitudes and moment values.
    :param m0_array: M0 values for all events that rupture a patch of interest
    :param mw_array: Moment magnitudes for all events that rupture a patch of interest
    :return:
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
    l3 = np.array([i for i in l1 for j in l2 if i == j])
    return np.unique(l3) # and i not in crossSec]

def mw_from_area_and_scaling_c(area: float, c: float):
    """
    Calculate the moment magnitude from an area and scaling parameter c.
    :param area: in m^2
    :param c: scaling parameter, usually 4.2 in NZ
    :return: moment magnitude
    """
    return np.log10(area) - 6. + c


def slip_from_area_and_scaling_c(area: float, c: float, mu: float =3.e10):
    """
    Calculate the slip from an area and scaling parameter c.
    :param area: in m^2
    :param c: scaling parameter, usually 4.2 in NZ
    :param mu: rigidity in Pa
    :return: slip in meters
    """

    m0 = mw_to_m0(mw_from_area_and_scaling_c(area, c))
    return m0 / (mu * area)

