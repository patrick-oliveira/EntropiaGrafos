import unittest as ut
from tests.src.math import TestEntropy
from tests.src.components import TestProbability, TestMemory

if __name__ == "__main__":
    suite = ut.TestSuite()
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestEntropy))
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestProbability))
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestMemory))
    r = ut.TextTestRunner()
    r.run(suite)
