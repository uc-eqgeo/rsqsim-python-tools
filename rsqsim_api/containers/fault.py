import numpy as np
from typing import Union, List
import os


def check_unique_vertices(vertex_array: np.ndarray, tolerance: Union[int, float] = 1):
    """

    :param vertex_array:
    :param tolerance:
    :return:
    """
    sorted_vertices = np.sort(vertex_array, axis=0)
    differences = sorted_vertices[1:] - sorted_vertices
    distances = np.linalg.norm(differences, axis=1)

    num_closer = len(distances[np.where(distances <= tolerance)])

    return num_closer, float(tolerance)


class RsqSimMultiFault:
    def __init__(self, faults: Union[list, tuple, set]):

        pass


    @classmethod
    def read_fault_file_charles(cls, fault_file: str):
        assert os.path.exists(fault_file)

        charles_dtype = [float] * 11 + [int] + ["U50"]
        charles_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                         "slip_rate", "fault_num", "fault_name"]

        data = np.genfromtxt(fault_file, dtype=charles_dtype, names=charles_names).T
        triangles = np.array([data[name] for name in charles_names[:9]])
        all_vertices = triangles.reshape(triangles.shape[0] * 3, int(triangles.shape[1] / 3))

        unique_vertices = np.unique(all_vertices, axis=0)

        num_closer, tolerance = check_unique_vertices(unique_vertices)
        if num_closer > 0:
            print("{:d} points are closer than {:.2f} m together: duplicates?". format(num_closer, tolerance))

        for i, triangle in enumerate(triangles):
            v1, v2, v3 = triangle.reshape(3, 3)



















class RsqSimGenericPatch:
    def __init__(self, fault: RsqSimFault, patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None):
        self._fault_name = None
        self._patch_number = None
        self._vertices = None
        self._dip_slip = None
        self._strike_slip = None

        self.fault = fault
        self.patch_number = patch_number
        self.dip_slip = dip_slip
        self.strike_slip = strike_slip

    #
    @property
    def fault_name(self):
        return self._fault_name

    @fault_name.setter
    def fault_name(self, name: str):
        if name is None:
            print("Warning: fault has no name.")
        # Check name doesn't have a stupid name
        self._fault_name = name

    # Patch number is zero if not specify
    @property
    def patch_number(self):
        return self._patch_number

    @patch_number.setter
    def patch_number(self, patch_number: int):
        assert patch_number >= 0, "Must be greater than zero!"
        self._patch_number = patch_number

    # I've written dip slip and strike slip as properties, in case we want to implement checks on values later
    @property
    def dip_slip(self):
        return self._dip_slip

    @dip_slip.setter
    def dip_slip(self, slip: float):
        self._dip_slip = slip

    @property
    def strike_slip(self):
        return self._strike_slip

    @strike_slip.setter
    def strike_slip(self, slip: float):
        self._strike_slip = slip

    @property
    def vertices(self):
        return self._vertices


class RsqSimTriangularPatch(RsqSimGenericPatch):
    """
    class to store information on an individual triangular patch of a fault
    """
    def __init__(self, fault: RsqSimFault, vertices: Union[list, np.ndarray, tuple], patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None):

        super(RsqSimTriangularPatch, self).__init__(fault=fault, patch_number=patch_number,
                                                    dip_slip=dip_slip, strike_slip=strike_slip)
        self.vertices = vertices

    @RsqSimGenericPatch.vertices.setter
    def vertices(self, vertices: Union[list, np.ndarray, tuple]):
        try:
            vertex_array = np.ndarray(vertices, dtype=np.float64)
        except ValueError:
            raise ValueError("Error parsing vertices for {}, patch {:d}".format(self.name, self.patch_number))

        assert vertex_array.ndim == 2, "2D array expected"
        # Check that at least 3 vertices supplied
        assert vertex_array.shape[0] >= 3, "At least 3 vertices (rows in array) expected"
        assert vertex_array.shape[1] == 3, "Three coordinates (x,y,z expected for each vertex"

        if vertex_array.shape[0] > 4:
            print("{}, patch {:d}: more patches than expected".format(self.name, self.patch_number))
            print("Taking first 3 vertices...")

        self._vertices = vertices[:3, :]

    @property
    def normal_vector(self):
        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[1] - self.vertices[2]
        cross_a_b = np.cross(a, b)
        # Ensure that normal always points up, normalize to give unit vector
        if cross_a_b[-1] < 0:
            unit_cross = -1 * cross_a_b / np.linalg.norm(cross_a_b)
        else:
            unit_cross = cross_a_b / np.linalg.norm(cross_a_b)
        return unit_cross

    @property
    def down_dip_vector(self):
        dx, dy, dz = self.normal_vector
        dd_vec = np.array(dx, dy, -1 / dz)
        return dd_vec / np.linalg.norm(dd_vec)

    @property
    def dip(self):
        horizontal = np.linalg.norm(self.down_dip_vector[:-1])
        vertical = -1 * self.down_dip_vector[:-1]
        return np.degrees(np.arctan(vertical / horizontal))
    
    @property
    def along_strike_vector(self):
        pass
    
    @property
    def centre(self):
        return np.mean(self.vertices, axis=0)

    @property
    def area(self):
        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[1] - self.vertices[2]
        area = 0.5 * np.linalg.norm(np.cross(a, b))
        return area


class RsqSimFault:
    def __init__(self):
        self._name = None
        self._patch_numbers = None
        self._patch_outlines = None
        self._vertices = None
        self._fault_number = None

    @property
    def name(self):
        return self._name

    @property
    def patch_numbers(self):
        return self._patch_numbers

    @property
    def patch_outlines(self):
        return self._patch_outlines

    @patch_outlines.setter
    def patch_outlines(self, triangles: List[Union[RsqSimTriangularPatch, RsqSimGenericPatch]]):



    def vertices(self):
        return self._vertices

    @classmethod
    def from_triangles(cls, fault_number: int, triangles: Union[np.ndarray, list, tuple],
                       patch_numbers: Union[list, tuple, set], fault_name: str = None):
        """

        :param fault_number:
        :param triangles:
        :param patch_numbers:
        :param fault_name:
        :return:
        """
        triangle_array = np.array(triangles)
        assert triangle_array.shape[1] == 9, "Expecting 3d coordinates of 3 vertices"
        assert len(patch_numbers) == triangle_array.shape[0], "Need one patch for each triangle"

        fault = cls()

        triangle_ls = []

        for patch_num, triangle in zip(patch_numbers, triangle_array):
            triangle3 = triangle.reshape(3, 3)
            patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num)
            triangle_ls.append(patch)



