import numpy as np
import pandas as pd
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

def mw_to_m0(magnitudes: np.ndarray):
    """
    Convert magnitudes to seismic moment.
    :param magnitudes:
    :return:
    """
    return 10 ** (1.5 * magnitudes + 9.05)

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

