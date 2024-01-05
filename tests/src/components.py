import unittest as ut
import numpy as np

from opdynamics.components.memory import (
    initialize_memory,
    probability_distribution
)
from opdynamics.components.utils import binary_to_int
from opdynamics.components.individual import Individual

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


class TestIndividual(ut.TestCase):
    distributions = [
        "binomial"
    ]
    kappa = 0
    memory_size = 32

    def setUp(self):
        self.expected_entropy = {
            "binomial": {
                "mean": 3.2,
                "std": 0.18
            }
        }

        self.expected_polarity = {
            "binomiial": {
                "mean": 0.5,
                "std": 0.03
            }
        }

    def test_individual_stats(self):
        for dist in self.distributions:
            Hs = []
            Pis = []

            for _ in range(10**5):
                ind = Individual(
                    kappa = self.kappa,
                    memory_size = self.memory_size,
                    distribution = dist
                )
                Hs.append(ind.H)
                Pis.append(ind.pi)

            H_mean = np.mean(Hs)
            H_std = np.std(Hs)
            expected_H_mean = self.expected_entropy[dist]["mean"]
            expected_H_std = self.expected_entropy[dist]["std"]

            Pi_mean = np.mean(Pis)
            Pi_std = np.std(Pis)
            expected_Pi_mean = self.expected_polarity[dist]["mean"]
            expected_Pi_std = self.expected_polarity[dist]["std"]

            with self.subTest(
                f"Individual mean entropy. Initial distribution: {dist}"
            ):
                self.assertAlmostEqual(
                    np.round(H_mean, 1),
                    expected_H_mean,
                    places=1
                )

                self.assertAlmostEqual(
                    np.round(H_std, 2),
                    expected_H_std,
                    places=1
                )

            with self.subTest(
                f"Individual mean polarity. Initial distribution: {dist}"
            ):
                self.assertAlmostEqual(
                    np.round(Pi_mean, 2),
                    expected_Pi_mean,
                    places=1
                )

                self.assertAlmostEqual(
                    np.round(Pi_std, 2),
                    expected_Pi_std,
                    places=1
                )
