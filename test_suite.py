import unittest as ut
from tests.src.math import TestPolarity
from tests.src.components import TestComponents
from tests.src.dynamics import TestDynamics
from tests.src.model import TestModel

if __name__ == "__main__":
    suite = ut.TestSuite()
    suite.addTest(ut.makeSuite(TestPolarity))
    suite.addTest(ut.makeSuite(TestComponents))
    suite.addTest(ut.makeSuite(TestDynamics))
    suite.addTest(ut.makeSuite(TestModel))

    r = ut.TextTestRunner()
    r.run(suite)
