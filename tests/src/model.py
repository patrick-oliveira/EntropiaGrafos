import unittest as ut
import numpy as np

from opdynamics.model.utils import order_indexes, group_indexes
from opdynamics.model.model import Model


class TestModel(ut.TestCase):
    def test_random_order(self):
        # Test that the function returns a random order of indices when
        # polarization_type is 0
        N = 10
        degrees = [2, 3, 1, 4, 5, 2, 1, 3, 4, 2]
        indices = order_indexes(N, 0, degrees)
        self.assertEqual(len(indices), N)
        self.assertNotEqual(indices, list(range(N)))

    def test_most_connected(self):
        # Test that the function returns an order of indices based on most
        # connected individuals when polarization_type is 1
        N = 10
        degrees = [2, 3, 1, 4, 5, 2, 1, 3, 4, 2]
        indices = order_indexes(N, 1, degrees)
        self.assertEqual(len(indices), N)
        self.assertEqual(indices, [4, 3, 8, 1, 0, 5, 9, 2, 6, 7])

    def test_less_connected(self):
        # Test that the function returns an order of indices based on less
        # connected individuals when polarization_type is 2
        N = 10
        degrees = [2, 3, 1, 4, 5, 2, 1, 3, 4, 2]
        indices = order_indexes(N, 2, degrees)
        self.assertEqual(len(indices), N)
        self.assertEqual(indices, [2, 6, 0, 5, 9, 7, 1, 8, 3, 4])

    def test_group_indexes(self):
        indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        alpha = 0.3
        omega = 0.2
        N = len(indexes)

        groups = group_indexes(indexes, alpha, omega, N)

        self.assertEqual(len(groups[1]), 3)
        self.assertEqual(len(groups[-1]), 2)
        self.assertEqual(len(groups[0]), 5)

        self.assertListEqual(groups[1], [1, 2, 3])
        self.assertListEqual(groups[-1], [4, 5])
        self.assertListEqual(groups[0], [6, 7, 8, 9, 10])

    def test_model(self):
        model_obj = Model(
            graph_type="barabasi",
            network_size=500,
            memory_size=256,
            code_length=5,
            kappa=0,
            lambd=0,
            alpha=0,
            omega=0,
            gamma=0,
            seed=42,
            preferential_attachment=2,
            polarization_type=0
        )

        model_entropy = np.round(model_obj.H, 1)
        model_polariz = np.round(model_obj.pi, 1)
        model_proximi = np.round(model_obj.S, 1)

        self.assertEqual(model_entropy, 3.5)
        self.assertEqual(model_polariz, 0.5)
        self.assertEqual(model_proximi, 1.0)

