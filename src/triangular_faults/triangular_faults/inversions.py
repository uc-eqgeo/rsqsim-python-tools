from typing import Union
import numpy as np
import numba


class InversionProblem:
    def __init__(self, design_matrix: np.ndarray, results_array: np.ndarray, weights_array: np.ndarray):
        assert all([isinstance(a, np.ndarray) for a in [design_matrix, results_array, weights_array]])
        assert design_matrix.ndim == 2
        assert all([a.ndim == 1 for a in [results_array, weights_array]])
        assert results_array.shape == weights_array.shape
        assert design_matrix.shape[0] == results_array.size
        self.design_matrix = design_matrix
        self.results_array = results_array
        self.weights_array = weights_array
        self.num_residuals = len(results_array)

    def fitness(self, x: np.ndarray):
        calc_disps = np.matmul(self.design_matrix, x)
        weighted_residuals = (calc_disps - self.results_array) * self.weights_array
        weighted_rms = np.mean(weighted_residuals**2)**0.5
        return np.array([weighted_rms])

    def get_bounds(self):
        num_params = self.design_matrix.shape[1]
        return [-20.] * num_params, [20.] * num_params





