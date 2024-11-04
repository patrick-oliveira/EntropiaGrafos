import unittest as ut
import numpy as np

from opdynamics.model.model import Model


class TestModel(ut.TestCase):
    distributions = [
        "binomial",
        "from_list"
    ]

    def setUp(self):
        np.random.seed(42)
        self.model_params = {
            "binomial": {
                "graph_type": "barabasi",
                "network_size": 500,
                "memory_size": 256,
                "code_length": 5,
                "kappa": 0,
                "lambd": 0,
                "alpha": 0,
                "omega": 0,
                "gamma": 0,
                "preferential_attachment": 2,
                "polarization_grouping_type": 0
            },
            "from_list": {
                "graph_type": "barabasi",
                "network_size": 500,
                "memory_size": 128,
                "code_length": 5,
                "kappa": 0,
                "lambd": 0,
                "alpha": 0,
                "omega": 0,
                "gamma": 0,
                "preferential_attachment": 2,
                "polarization_grouping_type": 0,
                "distribution": "from_list",
                "base_list": [0, 5, 10]
            }
        }

        self.expected_initial_values = {
            "binomial": {
                "entropy": 3.5,
                "proximity": 1.0,
                "polarity": 0.5
            },
            "from_list": {
                "entropy": 1.6,
                "proximity": 1.0,
                "polarity": 0.3
            }
        }

    def test_model_instantiation(self):
        for dist in self.distributions:
            model = Model(**self.model_params[dist])

            model_H = np.round(model.H, 1)
            model_S = np.round(model.J, 1)
            model_pi = np.round(model.pi, 1)

            expected_H = self.expected_initial_values[dist]["entropy"]
            expected_S = self.expected_initial_values[dist]["proximity"]
            expected_pi = self.expected_initial_values[dist]["polarity"]

            with self.subTest("Model entropy"):
                self.assertAlmostEqual(model_H, expected_H, 1)

            with self.subTest("Model proximity"):
                self.assertAlmostEqual(model_S, expected_S, 1)

            with self.subTest("Model polarity"):
                self.assertAlmostEqual(model_pi, expected_pi, 1)
