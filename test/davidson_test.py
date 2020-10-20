import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import tools 

from fermicluster import *

N = 1000
np.random.seed(2)
A = np.random.random((N,N))-np.eye(N)*.5
A = A + A.T


def test1():
    e1,U1 = np.linalg.eigh(A)
    idx = e1.argsort()
    e1 = e1[idx]
    U1 = U1[:,idx]
    print(e1[0])
    
    
    dav = Davidson(N, 1)
    dav.thresh      = 1e-12 
    dav.max_vecs    = 100 
    dav.max_iter    = 200 
    dav.form_rand_guess()
    for dit in range(0,dav.max_iter):
       
        dav.sig_curr = np.zeros((N, 1))
    
        dav.sig_curr = A @ dav.vec_curr
                
        dav.update()
        dav.print_iteration()
        if dav.converged():
            break
    if dav.converged():
        print(" Davidson Converged")
    else:
        print(" Davidson Not Converged")
    print() 
           
    e = dav.eigenvalues()
    v = dav.eigenvectors()
    
    print(" Eigenvalues of CI matrix:")
    print(" Davidson:     %18.14f"%(e[0]))
    print(" SciPy   :     %18.14f"%(e1[0]))

    assert(abs(e[0] - e1[0])<1e-12)

    ovlp = U1[:,0].T @ v
    print(" Overlap between states:", ovlp)
    assert(abs(abs(ovlp)-1)<1e-12)

if __name__== "__main__":
    test1() 
