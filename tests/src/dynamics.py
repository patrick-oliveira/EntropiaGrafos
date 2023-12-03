import unittest
import numpy as np
from opdynamics import SEED
from opdynamics.components.individual import Individual
from opdynamics.model.dynamics import (evaluate_information,
                                       get_transition_probabilities,
                                       distort,
                                       mutate,
                                       proximity)


class TestDynamics(unittest.TestCase):
    def test_acceptance_probability_1(self):
        # Test with acceptance probability of 1
        code = np.array([0, 1, 0, 1])
        result = evaluate_information(code, 1)
        self.assertTrue(np.array_equal(code, result))

    def test_acceptance_probability_0(self):
        # Test with acceptance probability of 0
        code = np.array([1, 0, 1, 0])
        result = evaluate_information(code, 0)
        self.assertIsNone(result)

    def test_acceptance_probability_05(self):
        # Test with acceptance probability of 0.5
        code = np.array([1, 1, 0, 0])
        result = evaluate_information(code, 0.5)
        self.assertIn(result, [None, code])

    def test_tendency_upwards(self):
        delta = 0.1
        xi = 0.2
        result = get_transition_probabilities(delta, xi, 1)
        expected = {0: 0.3, 1: 0.1}
        self.assertEqual(result, expected)

    def test_tendency_downwards(self):
        delta = 0.1
        xi = 0.2
        result = get_transition_probabilities(delta, xi, -1)
        expected = {0: 0.1, 1: 0.3}
        self.assertEqual(result, expected)

    def test_tendency_none(self):
        delta = 0.1
        xi = 0.2
        result = get_transition_probabilities(delta, xi)
        expected = {0: 0.1, 1: 0.1}
        self.assertEqual(result, expected)

    def test_distort(self):
        # Test 1: Test with all 0's
        code = np.zeros(10)
        transition_probability = get_transition_probabilities(0.4, 0.2, 1)
        distorted_code = distort(code, transition_probability)
        self.assertFalse(np.array_equal(distorted_code, np.zeros(10)))

        # Test 2: Test with all 1's
        code = np.ones(10)
        transition_probability = get_transition_probabilities(0.1, 0.4, -1)
        distorted_code = distort(code, transition_probability)
        self.assertFalse(np.array_equal(distorted_code, np.ones(10)))

        # Test 3: Test with random binary code
        code = np.random.randint(2, size=10)
        transition_probability = get_transition_probabilities(0.5, 0, None)
        distorted_code = distort(code, transition_probability)
        self.assertFalse(np.array_equal(distorted_code, code))

    def test_mutation(self):
        transition_probability = {
            0: 0.3,
            1: 0.1
        }

        bit_0 = 0
        bit_1 = 1
        mutations_0 = []
        mutations_1 = []
        for i in range(100000):
            bit_0_m = mutate(bit_0, transition_probability)
            bit_1_m = mutate(bit_1, transition_probability)
            if bit_0 != bit_0_m:
                mutations_0.append(1)
            else:
                mutations_0.append(0)

            if bit_1 != bit_1_m:
                mutations_1.append(1)
            else:
                mutations_1.append(0)

        mean_mutations_0 = np.round(np.mean(mutations_0), 1)
        mean_mutations_1 = np.round(np.mean(mutations_1), 1)
        self.assertEqual(mean_mutations_0, 0.3)
        self.assertEqual(mean_mutations_1, 0.1)

    def test_proximity(self):
        u = Individual(
            kappa=0,
            memory_size=256,
            code_length=5,
            distribution="binomial",
            seed=SEED
        )
        v = Individual(
            kappa=0,
            memory_size=256,
            code_length=5,
            distribution="poisson",
            lam=10,
            seed=SEED + 1
        )

        prox = proximity(u, v)
        prox = np.round(prox, 2)

        self.assertEqual(prox, 0.54)
