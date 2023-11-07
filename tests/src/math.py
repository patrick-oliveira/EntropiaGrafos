import unittest as ut
import numpy as np
from opdynamics.math.polarity import polarity_weights, polarity


class TestPolarity(ut.TestCase):
    def test_polarity_weights(self):
        code_length = 5
        weights = polarity_weights(code_length)
        
        self.assertTrue(isinstance(weights, np.ndarray))
        self.assertTrue(len(weights) == code_length)
        self.assertTrue(np.isclose(weights.sum(), 1.0))
        
    def test_polarity(self):
        # Test case 1: All bits are 0
        x = [0, 0, 0, 0]
        beta = [1, 2, 3, 4]
        self.assertEqual(polarity(x, beta), 0.0)

        # Test case 2: All bits are 1
        x = [1, 1, 1, 1]
        beta = [1, 2, 3, 4]
        self.assertEqual(polarity(x, beta), 10)

        # Test case 3: Random binary code and weight vector
        x = [1, 0, 1, 0]
        beta = [4, 3, 2, 1]
        self.assertEqual(polarity(x, beta), 6)
    