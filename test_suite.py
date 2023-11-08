import unittest as ut
from tests.src.math import TestPolarity
from tests.src.components import TestComponents

if __name__ == "__main__":
    suite = ut.TestSuite()
    suite.addTest(ut.makeSuite(TestPolarity))
    suite.addTest(ut.makeSuite(TestComponents))

    r = ut.TextTestRunner()
    r.run(suite)
