import unittest as ut
import numpy as np
from opdynamics import SEED
from opdynamics.components.individual import Individual
from opdynamics.components.memory import (generate_code,
                                          get_binary_codes,
                                          initialize_memory,
                                          probability_distribution)
from opdynamics.components.utils import (to_bin,
                                         to_int,
                                         complete_zeros,
                                         string_to_binary,
                                         binary_to_int,
                                         binary_to_string,
                                         random_selection)


class TestComponents(ut.TestCase):
    def test_to_bin(self):
        self.assertEqual(to_bin(0), '0')
        self.assertEqual(to_bin(1), '1')
        self.assertEqual(to_bin(2), '10')
        self.assertEqual(to_bin(10), '1010')
        self.assertEqual(to_bin(255), '11111111')

    def test_to_int(self):
        self.assertEqual(to_int('0'), 0)
        self.assertEqual(to_int('1'), 1)
        self.assertEqual(to_int('10'), 2)
        self.assertEqual(to_int('11'), 3)
        self.assertEqual(to_int('1010'), 10)
        self.assertEqual(to_int('1111'), 15)
        self.assertEqual(to_int('10000000'), 128)

    def test_complete_zeros(self):
        self.assertEqual(complete_zeros('101', 5), '00101')
        self.assertEqual(complete_zeros('1101', 3), '1101')
        self.assertEqual(complete_zeros('0', 2), '00')

    def test_string_to_binary(self):
        # Test case 1
        input_str = "110101"
        expected_output = [1, 1, 0, 1, 0, 1]
        self.assertEqual(string_to_binary(input_str).tolist(), expected_output)

        # Test case 2
        input_str = "01010101"
        expected_output = [0, 1, 0, 1, 0, 1, 0, 1]
        self.assertEqual(string_to_binary(input_str).tolist(), expected_output)

        # Test case 3
        input_str = "11111111"
        expected_output = [1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(string_to_binary(input_str).tolist(), expected_output)

    def test_binary_to_int(self):
        # Test binary number 0
        x = np.array([0, 0, 0, 0, 0])
        self.assertEqual(binary_to_int(x), 0)

        # Test binary number 1
        x = np.array([0, 0, 0, 0, 1])
        self.assertEqual(binary_to_int(x), 1)

        # Test binary number 10
        x = np.array([0, 0, 0, 1, 0])
        self.assertEqual(binary_to_int(x), 2)

        # Test binary number 11
        x = np.array([0, 0, 0, 1, 1])
        self.assertEqual(binary_to_int(x), 3)

        # Test binary number 101
        x = np.array([0, 0, 1, 0, 1])
        self.assertEqual(binary_to_int(x), 5)

        # Test binary number 1111
        x = np.array([0, 1, 1, 1, 1])
        self.assertEqual(binary_to_int(x), 15)

    def test_binary_to_string(self):
        x = np.array([0, 0, 1, 0, 1], dtype=np.int8)
        self.assertEqual(binary_to_string(x), "00101")

    def test_generate_code(self):
        x = 10
        m = 8
        expected_output = [0, 0, 0, 0, 1, 0, 1, 0]
        self.assertEqual(list(generate_code(x, m)), expected_output)

    def test_get_binary_codes(self):
        codes = get_binary_codes(256, 5, "binomial", SEED)
        numbers = list(map(binary_to_int, codes))
        mean = np.mean(numbers)

        self.assertAlmostEqual(mean, 15.8671865, places=2)
        codes = get_binary_codes(256, 5, "poisson", SEED, lam=1)
        numbers = list(map(binary_to_int, codes))
        mean = np.mean(numbers)

        self.assertAlmostEqual(mean, 0.9453125, places=2)
        codes = get_binary_codes(
            256,
            5,
            "from_list",
            SEED,
            base_list=[1, 2, 3]
        )
        numbers = list(map(binary_to_int, codes))
        mean = np.mean(numbers)
        self.assertAlmostEqual(mean, 1.93359375, places=2)

    def test_initialize_memory(self):
        codes, polarity = initialize_memory(256, 5, "binomial", SEED)

        numbers = list(map(binary_to_int, codes))
        mean = np.mean(numbers)
        self.assertAlmostEqual(mean, 15.8671865, places=2)

        mean = np.mean(polarity)
        self.assertAlmostEqual(mean, 0.5, places=2)

    def test_probability_distribution(self):
        codes, polarity = initialize_memory(256, 5, "binomial", SEED)
        dist = probability_distribution(codes, 256, 5)

        probs = dist.values()
        self.assertEqual(np.round(sum(probs), 1), 1.0)

    def test_random_selection(self):
        codes, polarity = initialize_memory(256, 5, "binomial", SEED)
        dist = probability_distribution(codes, 256, 5)

        random_codes = []
        for _ in range(100000):
            random_codes.append(binary_to_int(random_selection(dist)))

        mean = np.mean(random_codes)

        self.assertAlmostEqual(mean, 15.8, places=0)

    def test_individual(self):
        ind = Individual(
            kappa=0,
            memory_size=256,
            code_length=5,
            distribution="binomial",
            seed=SEED
        )

        ind_H = ind.H
        ind_delta = ind.delta
        ind_pi = ind.pi

        self.assertAlmostEqual(ind_H, 3.50606095, places=2)
        self.assertAlmostEqual(ind_delta, 0.5, places=2)
        self.assertAlmostEqual(ind_pi, 0.5046875, places=2)
