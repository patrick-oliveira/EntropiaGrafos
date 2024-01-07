import unittest as ut
from tests.src.math import TestEntropy
from tests.src.components import (
    TestProbability,
    TestMemory,
    TestIndividual,
)
from tests.src.model import (
    TestModel
)

if __name__ == "__main__":
    suite = ut.TestSuite()
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestEntropy))
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestProbability))
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestMemory))
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestIndividual))
    suite.addTest(ut.TestLoader().loadTestsFromTestCase(TestModel))
    r = ut.TextTestRunner()
    r.run(suite)
