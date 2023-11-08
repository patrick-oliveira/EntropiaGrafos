import unittest as ut
import numpy as np
from opdynamics import SEED
from opdynamics.math.polarity import polarity_weights, polarity
from opdynamics.math.entropy import shannon_entropy
from opdynamics.components.memory import get_binary_codes


class TestPolarity(ut.TestCase):
    def test_polarity_weights(self):
        code_length = 5
        weights = polarity_weights(code_length, SEED)

        self.assertTrue(isinstance(weights, np.ndarray))
        self.assertTrue(len(weights) == code_length)
        self.assertTrue(np.isclose(weights.sum(), 1.0))

    def test_polarity(self):
        x = [0, 0, 0, 0, 0]
        self.assertEqual(polarity(x, 5, SEED), 0.0)

        x = [1, 1, 1, 1]
        self.assertEqual(polarity(x, 4, SEED), 1)

        x = [1, 0, 1, 0]
        self.assertEqual(polarity(x, 4, SEED), 0.5)

        codes = get_binary_codes(256, 5, "binomial", SEED)
        self.assertAlmostEqual(
            np.mean(polarity(codes, 5, SEED)),
            0.5,
            places=2
        )

    def test_shannon_entropy(self):
        self.assertAlmostEqual(shannon_entropy(0.5), -0.5, places=2)
        self.assertAlmostEqual(shannon_entropy(0.25), -0.5, places=2)
        self.assertAlmostEqual(shannon_entropy(0.75), -0.31, places=2)
