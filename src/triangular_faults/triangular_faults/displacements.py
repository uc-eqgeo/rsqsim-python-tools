import numpy as np


class DisplacementArray:
    def __init__(self, x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray = None,
                 e_array: np.ndarray = None, n_array: np.ndarray = None, v_array: np.ndarray = None):
        assert x_array.shape == y_array.shape, "X and Y arrays should be the same size"
        assert x_array.ndim == 1, "Expecting 1D arrays"
        assert not all([a is None for a in [e_array, n_array, v_array]]), "Read in at least one set of displacements"

        self.x, self.y = x_array, y_array
        if z_array is None:
            self.z = np.zeros(self.x.shape)
        else:
            assert isinstance(z_array, np.ndarray)
            assert z_array.shape == self.x.shape
            self.z = z_array

        if e_array is not None:
            assert isinstance(e_array, np.ndarray)
            assert e_array.shape == self.x.shape
        self.e = e_array

        if n_array is not None:
            assert isinstance(n_array, np.ndarray)
            assert n_array.shape == self.x.shape
        self.n = n_array

        if v_array is not None:
            assert isinstance(v_array, np.ndarray)
            assert v_array.shape == self.x.shape
        self.v = v_array
