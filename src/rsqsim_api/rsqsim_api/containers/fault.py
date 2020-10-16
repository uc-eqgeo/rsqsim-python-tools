import glob
import os
from collections import Iterable
from typing import Union, List

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon
from tde.tde import calc_tri_displacements
from triangular_faults.displacements import DisplacementArray
from triangular_faults.utilities import read_ts_coords

transformer_utm2nztm = Transformer.from_crs(32759, 2193, always_xy=True)


def check_unique_vertices(vertex_array: np.ndarray, tolerance: Union[int, float] = 1):
    """
    Efficiently checks whether vertices (3D point coordinates) may be duplicates by (1) sorting them,
    ans (2) finding distances between adjacent points in the sorted array
    :param vertex_array: Numpy array with 3 columns representing 3D vertex coordinates
    :param tolerance: distance (in metres) below which points are reported as possible duplicates
    :return:
    """
    assert isinstance(vertex_array, np.ndarray)

    assert vertex_array.shape[1] == 3
    sorted_vertices = np.sort(vertex_array, axis=0)
    differences = sorted_vertices[1:] - sorted_vertices
    distances = np.linalg.norm(differences, axis=1)

    num_closer = len(distances[np.where(distances <= tolerance)])

    return num_closer, float(tolerance)


def normalize_bearing(bearing: Union[float, int]):
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing


class RsqSimMultiFault:
    def __init__(self, faults: Union[list, tuple, set]):
        self._faults = None

        self.faults = faults
        self.patch_dic = {}
        for fault in self.faults:
            if self.patch_dic is not None:
                self.patch_dic.update(fault.patch_dic)

        self.faults_with_patches = {patch_num: patch.segment for patch_num, patch in self.patch_dic.items()}

    def filter_faults_by_patch_numbers(self, patch_ls: Union[int, list, tuple, np.ndarray]):
        if isinstance(patch_ls, int):
            return self.patch_dic[patch_ls]
        else:
            assert isinstance(patch_ls, (tuple, list, np.ndarray))
            assert patch_ls
            assert all([isinstance(x, int) for x in patch_ls])
            fault_ls = list(set([self.patch_dic[patch_number].segment for patch_number in patch_ls]))
            return RsqSimMultiFault(fault_ls)

    @property
    def faults(self):
        return self._faults

    @faults.setter
    def faults(self, faults: Union[list, tuple, set]):
        assert all([isinstance(fault, (RsqSimMultiFault, RsqSimSegment)) for fault in faults])
        self._faults = faults

    @classmethod
    def read_fault_file(cls, fault_file: str, verbose: bool = False):
        """
        Read in opensha xml file?
        TODO: decide whether this is necessary, and if so finish it
        :param fault_file:
        :param verbose:
        :return:
        """
        assert os.path.exists(fault_file)
        # Read first 10 lines of file

    @classmethod
    def from_fault_file_keith(cls, fault_file: str, verbose: bool = False):
        """
        Read in an RSQSim fault file written according to Keith Richards-Dinger's convention.
        :param fault_file: Path to fault file
        :param verbose: Spit out more info if desired
        :return:
        """
        assert os.path.exists(fault_file)

        # Prepare info (types and headers) about columns
        column_dtypes = [float] * 11 + [int] + ["U50"]
        column_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                        "slip_rate", "fault_num", "fault_name"]

        # Read in data
        data = np.genfromtxt(fault_file, dtype=column_dtypes, names=column_names).T
        all_fault_nums = np.unique(data["fault_num"])
        num_faults = len(all_fault_nums)
        all_fault_names = np.unique(data["fault_name"])

        if not len(all_fault_names) == num_faults:
            print("Warning: not every fault has a corresponding name")

        # Populate faults with triangular patches
        patch_start = 0
        segment_ls = []

        for number in all_fault_nums:
            fault_data = data[data["fault_num"] == number]
            # Check that fault number has only one name associated with it
            associated_names = np.unique(fault_data["fault_name"])

            fault_name = associated_names[0] if associated_names.size > 0 else None

            if len(associated_names) > 1:
                print("More than one name provided for fault {:d}".format(number))
                print("Proceeding with first in list: {}".format(fault_name))
            if verbose:
                print("Reading fault: {:d}/{:d}: {}".format(number, num_faults, fault_name))

            # Extract data relevant for making triangles
            triangles = np.array([fault_data[name] for name in column_names[:9]]).T
            num_triangles = triangles.shape[0]
            # Reshape for creation of triangular patches
            all_vertices = triangles.reshape(triangles.shape[0] * 3, int(triangles.shape[1] / 3))

            patch_numbers = np.arange(patch_start, patch_start + num_triangles)

            if verbose:
                unique_vertices = np.unique(all_vertices, axis=0)
                num_closer, tolerance = check_unique_vertices(unique_vertices)
                if num_closer > 0:
                    print("{:d} points are closer than {:.2f} m together: duplicates?".format(num_closer, tolerance))

            # Create fault object
            fault_i = RsqSimSegment.from_triangles(triangles=triangles, patch_numbers=patch_numbers,
                                                   segment_number=number, fault_name=fault_name)

            segment_ls.append(fault_i)
            patch_start += num_triangles

        multi_fault = cls(segment_ls)

        return multi_fault

    @classmethod
    def read_fault_file_bruce(cls, main_fault_file: str, name_file: str, transform_from_utm: bool = False):
        assert all([os.path.exists(fname) for fname in (main_fault_file, name_file)])
        fault_names_messy = np.genfromtxt(name_file, dtype="U50")
        fault_names = []
        for name in fault_names_messy[:, 0]:
            fault_names.append("".join([char for char in name if char.isalnum()]))

        # Prepare info (types and headers) about columns
        column_dtypes = [float] * 11 + [int] + ["U50"]
        column_names = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "rake",
                        "slip_rate", "fault_num", "bruce_name"]

        # Read in data
        data = np.genfromtxt(main_fault_file, dtype=column_dtypes, names=column_names).T
        all_fault_df = pd.DataFrame(data)

        fault_names_set = set(fault_names)
        fault_num_set = set(data["fault_num"])
        fault_names_unique = []
        for name in fault_names:
            if name not in fault_names_unique:
                fault_names_unique.append(name)

        assert len(fault_names_set) == len(fault_num_set)

        # Populate faults with triangular patches
        patch_start = 0
        segment_ls = []

        for fault_num, fault_name in zip(fault_num_set, fault_names_unique):
            fault_data = all_fault_df[all_fault_df.fault_num == fault_num]

            num_triangles = len(fault_data)
            patch_numbers = np.arange(patch_start, patch_start + num_triangles)

            fault_i = RsqSimSegment.from_pandas(fault_data, fault_num, patch_numbers, fault_name,
                                                transform_from_utm=transform_from_utm)
            segment_ls.append(fault_i)
            patch_start += num_triangles

        multi_fault = cls(segment_ls)

        return multi_fault

    @classmethod
    def read_cfm_directory(cls, directory: str = None, files: Union[str, list, tuple] = None, shapefile=None):
        # TODO: finish writing
        directory_vs_file = [directory is None, files is None]
        assert not all(directory_vs_file)
        assert any(directory_vs_file)

        # Turn input of whatever format into list of ts files to read.
        if directory is not None:
            assert isinstance(directory, str)
            assert os.path.exists(directory)
            if directory[-1] != "/":
                ts_dir = directory + "/"
            else:
                ts_dir = directory

            ts_list = glob.glob(ts_dir + "*.ts")

        else:
            if isinstance(files, str):
                ts_list = [files]
            else:
                assert all([isinstance(fname, str) for fname in files])
                ts_list = list(files)

        assert len(ts_list) > 0, "No .ts files found..."

        for ts_file in ts_list:
            # Get fault name for comparison with shapefile
            ts_no_path = os.path.basename(ts_file)
            ts_name = ts_no_path.split(".ts")[0]


class RsqSimSegment:
    def __init__(self, segment_number: int, patch_type: str = "triangle", fault_name: str = None):
        """

        :param segment_number:
        :param patch_type:
        :param fault_name:
        """
        self._name = None
        self._patch_numbers = None
        self._patch_outlines = None
        self._patch_vertices = None
        self._vertices = None
        self._triangles = None
        self._edge_lines = None
        self._segment_number = segment_number
        self._patch_type = None
        self._adjacency_map = None
        self._laplacian = None

        self.patch_type = patch_type
        self.name = fault_name
        self.ss_gf, self.ds_gf = (None,) * 2

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, fault_name: str):
        if fault_name is None:
            self._name = None
        else:
            assert isinstance(fault_name, str)
            assert " " not in fault_name, "No spaces in fault name, please..."
            self._name = fault_name.lower()

    @property
    def patch_numbers(self):
        return self._patch_numbers

    @patch_numbers.setter
    def patch_numbers(self, numbers: Union[list, tuple, np.ndarray]):
        number_array = np.array(numbers)
        assert number_array.dtype == "int"
        if self.patch_outlines is not None:
            assert len(number_array) == len(self.patch_outlines)
        self._patch_numbers = number_array

    @property
    def segment_number(self):
        return self._segment_number

    @property
    def patch_type(self):
        return self._patch_type

    @patch_type.setter
    def patch_type(self, patch_type: str):
        assert isinstance(patch_type, str)
        patch_lower = patch_type.lower()
        assert patch_lower in ("triangle", "rectangle", "tri", "rect"), "Expecting 'triangle' or 'rectangle'"
        if patch_lower in ("triangle", "tri"):
            self._patch_type = "triangle"
        else:
            self._patch_type = "rectangle"

    @property
    def patch_outlines(self):
        return self._patch_outlines

    @property
    def patch_vertices(self):
        return self._patch_vertices

    @patch_outlines.setter
    def patch_outlines(self, patches: List):
        if self.patch_type == "triangle":
            assert all([isinstance(patch, RsqSimTriangularPatch) for patch in patches])
        elif self.patch_type == "rectangle":
            assert all([isinstance(patch, RsqSimGenericPatch) for patch in patches])
        else:
            raise ValueError("Set patch type (triangle or rectangle) for fault!")

        self._patch_outlines = patches
        self._patch_vertices = [patch.vertices for patch in patches]

    @property
    def vertices(self):
        if self._vertices is None:
            self.get_unique_vertices()
        return self._vertices

    def get_unique_vertices(self):
        if self.patch_vertices is None:
            raise ValueError("Read in triangles first!")
        all_vertices = np.reshape(self.patch_vertices, (3 * len(self.patch_vertices), 3))
        unique_vertices = np.unique(all_vertices, axis=0)
        self._vertices = unique_vertices

    @property
    def triangles(self):
        if self._triangles is None:
            self.generate_triangles()
        return self._triangles

    @property
    def edge_lines(self):
        if self._edge_lines is None:
            self.generate_triangles()
        return self._edge_lines

    def generate_triangles(self):
        assert self.patch_outlines is not None, "Load patches first!"
        all_vertices = [patch.vertices for patch in self.patch_outlines]
        unique_vertices = np.unique(np.vstack(all_vertices), axis=0)
        self._vertices = unique_vertices

        triangle_ls = []
        line_ls = []
        for triangle in all_vertices:
            vertex_numbers = []
            for vertex in triangle:
                index = np.where((unique_vertices == vertex).all(axis=1))[0][0]
                vertex_numbers.append(index)
            triangle_ls.append(vertex_numbers)
            line_ls += [[vertex_numbers[0], vertex_numbers[1]],
                        [vertex_numbers[0], vertex_numbers[2]],
                        [vertex_numbers[1], vertex_numbers[2]]]
        self._triangles = np.array(triangle_ls)
        self._edge_lines = np.array(line_ls)

    def find_triangles_from_vertex_index(self, vertex_index: int):
        assert isinstance(vertex_index, int)
        assert 0 <= vertex_index < len(self.vertices)
        triangle_index_list = []
        for i, triangle in enumerate(self.triangles):
            if vertex_index in triangle:
                triangle_index_list.append(i)

        print(triangle_index_list)
        return triangle_index_list

    @classmethod
    def from_triangles(cls, triangles: Union[np.ndarray, list, tuple], segment_number: int = 0,
                       patch_numbers: Union[list, tuple, set, np.ndarray] = None, fault_name: str = None,
                       strike_slip: Union[int, float] = None, dip_slip: Union[int, float] = None):
        """
        Create a segment from triangle vertices and (if appropriate) populate it with strike-slip/dip-slip values
        :param segment_number:
        :param triangles:
        :param patch_numbers:
        :param fault_name:
        :param strike_slip:
        :param dip_slip:
        :return:
        """
        # Test shape of input array is appropriate
        triangle_array = np.array(triangles)
        assert triangle_array.shape[1] == 9, "Expecting 3d coordinates of 3 vertices each"
        if patch_numbers is None:
            patch_numbers = np.arange(len(triangle_array))
        else:
            assert len(patch_numbers) == triangle_array.shape[0], "Need one patch for each triangle"

        # Create empty segment object
        fault = cls(patch_type="triangle", segment_number=segment_number, fault_name=fault_name)

        triangle_ls = []

        # Populate segment object
        for patch_num, triangle in zip(patch_numbers, triangle_array):
            triangle3 = triangle.reshape(3, 3)
            patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num, strike_slip=strike_slip,
                                          dip_slip=dip_slip)
            triangle_ls.append(patch)

        fault.patch_outlines = triangle_ls
        fault.patch_numbers = np.array([patch.patch_number for patch in triangle_ls])
        fault.patch_dic = {p_num: patch for p_num, patch in zip(fault.patch_numbers, fault.patch_outlines)}

        return fault

    @classmethod
    def from_tsurface(cls, tsurface_file: str):
        assert os.path.exists(tsurface_file)
        _, _, tri = read_ts_coords(tsurface_file)
        fault = cls.from_triangles(tri)
        return fault

    @classmethod
    def from_pandas(cls, dataframe: pd.DataFrame, segment_number: int,
                    patch_numbers: Union[list, tuple, set, np.ndarray], fault_name: str = None,
                    strike_slip: Union[int, float] = None, dip_slip: Union[int, float] = None,
                    transform_from_utm: bool = False):

        triangles = dataframe.iloc[:, :9].to_numpy()
        if transform_from_utm:
            reshaped_array = triangles.reshape((len(triangles) * 3), 3)
            transformed_array = transformer_utm2nztm.transform(reshaped_array[:, 0], reshaped_array[:, 1],
                                                               reshaped_array[:, 2])
            reordered_array = np.vstack(transformed_array).T
            triangles_nztm = reordered_array.reshape((len(triangles), 9))

        else:
            triangles_nztm = triangles

        # Create empty segment object
        fault = cls(patch_type="triangle", segment_number=segment_number, fault_name=fault_name)

        triangle_ls = []

        # Populate segment object
        for patch_num, triangle in zip(patch_numbers, triangles_nztm):
            triangle3 = triangle.reshape(3, 3)
            patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num,
                                          strike_slip=strike_slip,
                                          dip_slip=dip_slip)
            triangle_ls.append(patch)

        fault.patch_outlines = triangle_ls
        fault.patch_numbers = np.array([patch.patch_number for patch in triangle_ls])
        fault.patch_dic = {p_num: patch for p_num, patch in zip(fault.patch_numbers, fault.patch_outlines)}

        return fault

    def collect_greens_function(self, sites_x: Union[list, tuple, np.ndarray], sites_y: Union[list, tuple, np.ndarray],
                                sites_z: Union[list, tuple, np.ndarray] = None, strike_slip: Union[int, float] = 1,
                                dip_slip: Union[int, float] = 1, poisson_ratio: Union[int, float] = 0.25,
                                tensional_slip: Union[int, float] = 0):
        """

        :param sites_x:
        :param sites_y:
        :param sites_z:
        :param strike_slip:
        :param dip_slip:
        :param poisson_ratio:
        :param tensional_slip:
        :return:
        """
        x_array = np.array(sites_x)
        y_array = np.array(sites_y)
        if sites_z is None:
            z_array = np.zeros(x_array.shape)
        else:
            z_array = np.array(sites_z)

        x_vertices, y_vertices, z_vertices = [[vertices.T[i] for vertices in self.patch_vertices] for i in range(3)]

        ds_gf_dic = {}
        ss_gf_dic = {}
        for patch_number, xv, yv, zv in zip(self.patch_numbers, x_vertices, y_vertices, z_vertices):
            ds_gf_dic[patch_number] = calc_tri_displacements(x_array, y_array, z_array, xv, yv, -1. * zv,
                                                             poisson_ratio, 0., tensional_slip, dip_slip)
            ss_gf_dic[patch_number] = calc_tri_displacements(x_array, y_array, z_array, xv, yv, -1. * zv,
                                                             poisson_ratio, strike_slip, tensional_slip, 0.)

        self.ds_gf = ds_gf_dic
        self.ss_gf = ss_gf_dic

    def gf_design_matrix(self, displacements: DisplacementArray):
        """
        Maybe this should be stored on the DisplacementArray object?
        :param displacements:
        :return:
        """
        assert isinstance(displacements, DisplacementArray), "Expecting DisplacementArray object"
        if any([a is None for a in (self.ds_gf, self.ss_gf)]):
            self.collect_greens_function(displacements.x, displacements.y, displacements.z)
        design_matrix_ls = []
        d_list = []
        w_list = []
        for component, key in zip([displacements.e, displacements.n, displacements.v], ["x", "y", "z"]):
            if component is not None:
                for site in range(len(component)):
                    component_ds = [self.ds_gf[i][key][site] for i in self.patch_numbers]
                    component_ss = [self.ss_gf[i][key][site] for i in self.patch_numbers]
                    component_combined = component_ds + component_ss
                    design_matrix_ls.append(component_combined)
                d_list.append(component)

        design_matrix_array = np.array(design_matrix_ls)
        d_array = np.hstack([a for a in d_list])

        return design_matrix_array, d_array

    @property
    def adjacency_map(self):
        if self._adjacency_map is None:
            self.build_adjacency_map()
        return self._adjacency_map

    def build_adjacency_map(self):
        '''
        For each triangle vertex, find the indices of the adjacent triangles.
        This function overwrites that from the parent class TriangularPatches.

        :Kwargs:
            * verbose       : Speak to me

        :Returns:
            * None
        '''

        self._adjacency_map = []

        # Cache the vertices and faces arrays

        # First find adjacent triangles for all triangles
        # Currently any triangle with a edge, could be a common vertex instead.
        for vertex_numbers in self.triangles:
            adjacent_triangles = []
            for j, triangle in enumerate(self.triangles):
                common_vertices = [a for a in vertex_numbers if a in triangle]
                if len(common_vertices) == 2:
                    adjacent_triangles.append(j)
            self._adjacency_map.append(adjacent_triangles)

    def build_laplacian_matrix(self):

        '''
        Build a discrete Laplacian smoothing matrix.

        :Args:
            * verbose       : if True, displays stuff.
            * method        : Method to estimate the Laplacian operator

                - 'count'   : The diagonal is 2-times the number of surrounding nodes. Off diagonals are -2/(number of surrounding nodes) for the surrounding nodes, 0 otherwise.
                - 'distance': Computes the scale-dependent operator based on Desbrun et al 1999. (Mathieu Desbrun, Mark Meyer, Peter Schr\"oder, and Alan Barr, 1999. Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow, Proceedings of SIGGRAPH).

            * irregular     : Not used, here for consistency purposes

        :Returns:
            * Laplacian     : 2D array
        '''

        # Build the tent adjacency map
        if self.adjacency_map is None:
            self.build_adjacency_map()

        # Get the vertices

        # Allocate an array
        laplacian_matrix = np.zeros((len(self.patch_numbers), len(self.patch_numbers)))

        # Normalize the distances
        all_distances = []
        for i, (patch, adjacents) in enumerate(zip(self.patch_outlines, self.adjacency_map)):
            patch_centre = patch.centre
            distances = np.array([np.linalg.norm(self.patch_outlines[a].centre - patch_centre) for a in adjacents])
            all_distances.append(distances)
        normalizer = np.max([np.max(d) for d in all_distances])

        # Iterate over the vertices
        for i, (adjacents, distances) in enumerate(zip(self.adjacency_map, all_distances)):
            # Distance-based
            distances_normalized = distances / normalizer
            e = np.sum(distances_normalized)
            laplacian_matrix[i, i] = float(len(adjacents)) * 2. / e * np.sum(1. / distances_normalized)
            laplacian_matrix[i, adjacents] = -2. / e * 1. / distances_normalized

        self._laplacian = np.hstack((laplacian_matrix, laplacian_matrix))

    @property
    def laplacian(self):
        if self._laplacian is None:
            self.build_laplacian_matrix()
        return self._laplacian

    def find_top_patches(self, depth_tolerance: Union[float, int] = 100):
        top_vertex_depth = max(self.vertices[:, -1])
        shallow_indices = np.where(self.vertices[:, -1] >= top_vertex_depth - depth_tolerance)[0]
        return shallow_indices


class RsqSimFault:
    """
    The idea is to allow a fault to have one or more segments
    """

    def __init__(self, segments: Union[RsqSimSegment, List[RsqSimSegment]]):
        self._segments = None
        self._vertices = None

        if segments is not None:
            self.segments = segments

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments: Union[RsqSimSegment, List[RsqSimSegment]]):

        if isinstance(segments, RsqSimSegment):
            self._segments = [segments]
        else:
            assert isinstance(segments, Iterable), "Expected either one segment or a list of segments"
            assert all([isinstance(segment, RsqSimSegment) for segment in segments]), "Expected a list of segments"
            self._segments = list(segments)


class RsqSimGenericPatch:
    def __init__(self, segment: RsqSimSegment, patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None):
        self._patch_number = None
        self._vertices = None
        self._dip_slip = None
        self._strike_slip = None

        self.segment = segment
        self.patch_number = patch_number
        self.dip_slip = dip_slip
        self.strike_slip = strike_slip

    # Patch number is zero if not specified
    @property
    def patch_number(self):
        return self._patch_number

    @patch_number.setter
    def patch_number(self, patch_number: np.integer):
        assert isinstance(patch_number, (np.integer, int)), "Patch number must be an integer"
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

    @property
    def total_slip(self):
        return np.linalg.norm(np.array([self.strike_slip, self.dip_slip]))


class RsqSimTriangularPatch(RsqSimGenericPatch):
    """
    class to store information on an individual triangular patch of a fault
    """

    def __init__(self, segment: RsqSimSegment, vertices: Union[list, np.ndarray, tuple], patch_number: int = 0,
                 dip_slip: float = None, strike_slip: float = None, rake: Union[int, float] = None,
                 ):

        super(RsqSimTriangularPatch, self).__init__(segment=segment, patch_number=patch_number,
                                                    dip_slip=dip_slip, strike_slip=strike_slip)
        self.vertices = vertices
        self._normal_vector = self.calculate_normal_vector()
        self._down_dip_vector = self.calculate_down_dip_vector()
        self._dip = self.calculate_dip()
        self._along_strike_vector = self.calculate_along_strike_vector()
        self._centre = self.calculate_centre()
        self._area = self.calculate_area()

    @RsqSimGenericPatch.vertices.setter
    def vertices(self, vertices: Union[list, np.ndarray, tuple]):
        try:
            vertex_array = np.array(vertices, dtype=np.float64)
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

    def calculate_normal_vector(self):
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
    def normal_vector(self):
        return self._normal_vector

    @property
    def down_dip_vector(self):
        return self._down_dip_vector

    def calculate_down_dip_vector(self):
        dx, dy, dz = self.normal_vector
        dd_vec = np.array([dx, dy, -1 / dz])
        return dd_vec / np.linalg.norm(dd_vec)

    @property
    def dip(self):
        return self._dip

    def calculate_dip(self):
        horizontal = np.linalg.norm(self.down_dip_vector[:-1])
        vertical = -1 * self.down_dip_vector[:-1]
        return np.degrees(np.arctan(vertical / horizontal))

    @property
    def along_strike_vector(self):
        return self._along_strike_vector

    @property
    def strike(self):
        if self.along_strike_vector is not None:
            strike = 90. - np.degrees(np.arctan2(self.along_strike_vector[1], self.along_strike_vector[0]))
            return normalize_bearing(strike)
        else:
            return None

    def calculate_along_strike_vector(self):
        return np.cross(self.normal_vector, self.down_dip_vector)

    @property
    def centre(self):
        return self._centre

    def calculate_centre(self):
        return np.mean(self.vertices, axis=0)

    @property
    def area(self):
        return self._area

    def calculate_area(self):
        a = self.vertices[1] - self.vertices[0]
        b = self.vertices[1] - self.vertices[2]
        area = 0.5 * np.linalg.norm(np.cross(a, b))
        return area

    def slip3d_to_ss_ds(self, x1_slip: Union[float, int], x2_slip: Union[float, int], x3_slip: Union[float, int]):
        sv_3d = np.array([x1_slip, x2_slip, x3_slip])
        ss = np.dot(self.along_strike_vector, sv_3d)
        ds = np.dot(-1 * self.down_dip_vector, sv_3d)

        return ds, ss

    def horizontal_sv_to_ds_ss(self, slipvec, magnitude: Union[float, int] = 1):
        """
        Program to perform the 'inverse' of slipvec.
        Arguments: strike, dip azimuth of slip vector (all degrees)
        Returns rake (degrees)
        """
        assert isinstance(slipvec, ())
        angle = self.strike - slipvec
        if angle < -180.:
            angle = 360. + angle
        elif angle > 180.:
            angle = angle - 360.

        if angle == 90.:
            strike_perp = magnitude
            strike_par = 0.
        elif angle == -90.:
            strike_perp = -1. * magnitude
            strike_par = 0
        else:
            strike_par = np.cos(np.radians(angle))
            strike_perp = np.cos(np.radians(angle)) / np.cos(np.radians(self.dip))
            normalizer = magnitude / np.linag.norm(np.array([strike_par, strike_perp]))
            strike_par *= normalizer
            strike_perp *= normalizer

        return strike_perp, strike_par

    def as_polygon(self):
        return Polygon(self.vertices)

    # def calculate_greens_functions(self, ):
