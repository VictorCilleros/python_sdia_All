import numpy as np
def gradient2D(X):
    """Computes the discrete gradient operator
     $D: \mathbb{C}^{M\times N} \rightarrow \mathbb{C}^{M\times N} \times \mathbb{C}^{M\times N} $
     $D(X) &= (\mathbf{XD}_h, \mathbf{D}_v\mathbf{X}) \in \mathbb{C}^{M\times N} \times \mathbb{C}^{M\times N}$
     Where M & N are X numpy.ndarray shape.

     Parameters:
     X :  ndarray  from numpy.ndarray in C^(M,N)
     Should not have a dimension greater than 2
     Returns:
     (A,B): tuple of ndarray in C^(M,N) x C^(M,N)
     the tuple containing the discrete gradient operator.
    """
    assert np.shape(X) != ()
    assert (np.shape(X)[0] > 1 and np.shape(X)[1] >1),"Input array has more than 2 dimensions"
    return( np.concatenate((X.T[1:]-X.T[:-1],[np.zeros(X.shape[0])])).T,
            np.concatenate((X[1:]-X[:-1],[np.zeros(X.shape[0])])))


def tv(X):
    """This function compute the discrete isotropic total variation of an input matrix in C^(M,N)
    Args:
    X (np.array): A matrix in C^(MxN)
    Returns:
    float: returns the value of the TV for the input matrix X
    """
    (Dh,Dv)=gradient2D(X)
    return np.sum(np.sqrt(np.square(Dh)+np.square(Dv)))


def gradient2D_adjoint(Y):
    """This function computes the adjoint of the 2D discrete gradient operator D applied to a couple of matrices of dimension 2
    Parameters:
    Y (array,array): A tuple in C^(M,N) x C^(M,N)
    Returns:
    array: A matrix in C^(M,N)
    """
    (Yh,Yv)=Y
    M=np.shape(Yh)[0]
    N=np.shape(Yh)[1]

    A=   np.hstack(  ( np.zeros((M,1)) , np.delete(Yh,[N-1,N-2],1)) )       # Avec les colonnes
    B=  np.hstack(  ( np.zeros((N,1)) , np.delete(Yv.T,[M-1,M-2],1)) )
    U=-Yh.T[:-1].T+A
    V=(-Yv[:-1].T+B)

    return np.hstack((U,np.array([Yh.T[-2]]).T))  + np.hstack((V,np.array([Yv[-2]]).T)).T
