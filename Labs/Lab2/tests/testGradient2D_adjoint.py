import unittest
from src import functions
import numpy as np

def scalar_product(A,B):
    return np.trace(A.T@B)
def scalar_product_2(A,B):
    (A1,A2)=A
    (B1,B2)=B
    return scalar_product(A1,B1)+scalar_product(A2,B2)


class Test_TestGradient2D_adjoint(unittest.TestCase):

    def test_gradient_adjoint(self):
        M,N = 2,2
        Generator1 = np.random.default_rng(10004)
        Generator2 = np.random.default_rng(10)
        Generator3 = np.random.default_rng(10544)
        X=Generator1.standard_normal(size=(M,N))
        Y = (Generator2.standard_normal(size=(M,N)),Generator3.standard_normal(size=(M,N))) # définir le sample

        D_x=functions.gradient2D(X)
        D_star_y=functions.gradient2D_adjoint(Y)
        assert np.abs(scalar_product_2(D_x,Y) - scalar_product(X,D_star_y)) <= 1e-6