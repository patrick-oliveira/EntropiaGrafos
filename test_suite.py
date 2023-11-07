import unittest as ut
from tests.src.math import TestPolarity

if __name__ == "__main__":
    suite = ut.TestSuite()
    suite.addTest(ut.makeSuite(TestPolarity))
    
    r = ut.TextTestRunner()
    r.run(suite)
    