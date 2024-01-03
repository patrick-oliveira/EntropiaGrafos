import unittest as ut
import numpy as np

from opdynamics.utils.types import (
    CodeDistribution
)
from opdynamics.math.entropy import (
    shannon_entropy,
    memory_entropy,
    JSD,
    S
)
from opdynamics.components.memory import (
    initialize_memory,
    probability_distribution
)


class TestEntropy(ut.TestCase):
    memory_size = 32

    def setUp(self):
        self.test_distribution = CodeDistribution(
            distribution = {
                '00000': 0.0,
                '00001': 0.0,
                '00010': 0.0,
                '00011': 0.0,
                '00100': 0.0,
                '00101': 0.0,
                '00110': 0.0,
                '00111': 0.0,
                '01000': 0.03125,
                '01001': 0.0,
                '01010': 0.0,
                '01011': 0.0,
                '01100': 0.0625,
                '01101': 0.0625,
                '01110': 0.125,
                '01111': 0.0625,
                '10000': 0.1875,
                '10001': 0.0625,
                '10010': 0.15625,
                '10011': 0.125,
                '10100': 0.0625,
                '10101': 0.0625,
                '10110': 0.0,
                '10111': 0.0,
                '11000': 0.0,
                '11001': 0.0,
                '11010': 0.0,
                '11011': 0.0,
                '11100': 0.0,
                '11101': 0.0,
                '11110': 0.0,
                '11111': 0.0
            }
        )
        self.expected_memory_entropy = 3.277518266288633

    def test_shannon_entropy(self):
        self.assertEqual(shannon_entropy(np.array([0])), 0)
        self.assertEqual(shannon_entropy(np.array([1])), 0)
        self.assertAlmostEqual(
            shannon_entropy(2**(-1 / np.log(2))),
            0.530737845423043,
            places=8
        )

    def test_memory_entropy(self):
        self.assertAlmostEqual(
            memory_entropy(self.test_distribution),
            self.expected_memory_entropy,
            places=5
        )

    def test_jsd(self):
        ma = initialize_memory(
            memory_size = self.memory_size,
            distribution = "binomial"
        )
        da = probability_distribution(
            memory = ma,
            memory_size = self.memory_size
        )

        with self.subTest("JSD sanity test."):
            self.assertEqual(
                JSD(da, da),
                0.0
            )

        with self.subTest("Proximity sanity test."):
            self.assertEqual(
                S(da, da),
                1.0
            )
