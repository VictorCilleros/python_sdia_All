import unittest
from src import functions
import numpy as np

class Test_TestGradient2D(unittest.TestCase):

    def test_gradient(self):
        c,n=2,2
        assert (functions.gradient2D(c*np.ones((n,n)))[0] == np.zeros(n)).all()

