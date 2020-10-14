import unittest
import numpy as np
from rsqsim_api.containers.fault import RsqSimTriangularPatch, RsqSimSegment
test_vertices = np.array([[0., 0., 0.],
                          [0., 1., 0.],
                          [0., 1., -1]])


class TestTriangularPatch(unittest.TestCase):
    def setUp(self) -> None:
        self.segment = RsqSimSegment(0)
        self.triangle = RsqSimTriangularPatch(segment=self.segment, vertices=test_vertices)

    def test_return_vertices_shape(self):
        self.assertSequenceEqual(self.triangle.vertices.shape, test_vertices.shape)

    def test_return_vertices(self):
        np.testing.assert_array_almost_equal(self.triangle.vertices, test_vertices)

    def test_normal_vector(self):
        np.testing.assert_array_almost_equal(self.triangle.normal_vector, np.array([1., 0., 0.]))

    def test_down_dip_vector(self):
        np.testing.assert_array_almost_equal(self.triangle.down_dip_vector, np.array([]))

    def test_dip_magnitude(self):
        np.testing.assert_almost_equal(self.triangle.dip)
