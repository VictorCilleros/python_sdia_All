import unittest
from src import functions
import numpy as np

class Test_TestTv(unittest.TestCase):

    def test_tv(self):
        c,n=2,10
        assert functions.tv(c*np.ones((n,n))) == 0