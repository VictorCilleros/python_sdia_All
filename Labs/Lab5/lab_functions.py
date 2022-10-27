import numpy as np


def markov(rho,A,nmax,rng=np.random):
    X=np.zeros(nmax,dtype=int) #on indique que X contiendra seulement des entiers.
    X[0]=rng.choice(a=range(len(rho)), p=rho)
    for k in range(nmax-1):
        X[k+1]=rng.choice(a=range(len(A)), p=A[X[k],:])
        # Les états sont numérotés de 0 à len(A)
    return X


def gen_transition(n):
    return(np.random.rand(n,n))

def gen_rho(n):
    return(np.random.rand(1,n))

def transition(rho,A,nmax):
    R = [rho]
    for i in range(1,nmax):
        R.append(R[-1]@A)
    return(R)