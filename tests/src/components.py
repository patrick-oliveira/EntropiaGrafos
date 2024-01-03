import unittest as ut
import numpy as np

from opdynamics.components.memory import (
    initialize_memory,
    probability_distribution
)
from opdynamics.components.utils import binary_to_int

np.random.seed(42)


class TestProbability(ut.TestCase):
    distributions = [
        "binomial"
    ]
    memory_size = 32

    def setUp(self):
        self.expected_means = {
            "binomial": 16
        }

    def test_sanity_test(self):
        for dist in self.distributions:
            memory = initialize_memory(
                memory_size = self.memory_size,
                distribution = dist
            )
            memory_distribution = probability_distribution(
                memory = memory,
                memory_size = 32
            ).distribution

            probabilities = np.array(list(memory_distribution.values()))

            with self.subTest(f"Sanity test for {dist} distribution."):
                self.assertAlmostEqual(
                    np.round(np.sum(probabilities), 2),
                    1.0,
                    places=1
                )

    def test_probability_distributions(self):
        for dist in self.distributions:
            probs_sum = np.zeros(self.memory_size)

            for _ in range(10**4):
                memory = initialize_memory(
                    memory_size = self.memory_size,
                    distribution = dist
                )
                memory_distribution = probability_distribution(
                    memory = memory,
                    memory_size = 32
                ).distribution

                probabilities = np.array(list(memory_distribution.values()))
                probs_sum += probabilities

            mean_probs = probs_sum / 10**4

            with self.subTest(f"Mean for {dist} distribution."):
                self.assertEqual(
                    np.argmax(mean_probs),
                    self.expected_means[dist]
                )


class TestMemory(ut.TestCase):
    distributions = [
        "binomial"
    ]
    memory_size = 16

    def setUp(self):
        self.expected_means = {
            "binomial": 16
        }
        self.expected_stds = {
            "binomial": 2.7
        }
        self.expected_polarity_mean = {
            "binomial": 0.5
        }

    def test_initialized_memory_dist(self):
        for dist in self.distributions:
            means = []
            stds = []
            for _ in range(10**4):
                memory = initialize_memory(
                    memory_size = self.memory_size,
                    distribution = dist
                )
                ints = [binary_to_int(x) for x in memory.codes]
                means.append(np.mean(ints))
                stds.append(np.std(ints))

            with self.subTest(f"Mean for {dist} distribution."):
                self.assertAlmostEqual(
                    np.round(np.mean(means), 1),
                    self.expected_means[dist],
                    places=1
                )

            with self.subTest(f"Standard deviation for {dist} distribution."):
                self.assertAlmostEqual(
                    np.round(np.mean(stds), 1),
                    self.expected_stds[dist],
                    places=1
                )

    def test_initialized_memory_polarity(self):
        for dist in self.distributions:
            means = []
            for _ in range(10**3):
                memory = initialize_memory(
                    memory_size = self.memory_size,
                    distribution = dist
                )
                means.append(np.mean(memory.polarities))

            with self.subTest(f"Polarity mean for {dist} distribution."):
                self.assertAlmostEqual(
                    np.round(np.mean(means), 2),
                    self.expected_polarity_mean[dist],
                    places=1
                )
