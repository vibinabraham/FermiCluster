import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from fermicluster import *
from pyscf_helper import *
import pyscf

def test_1():
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      0.00       0.10       1.50
    H      1.10       0.10       1.50
    H      0.00       0.20       2.50
    H      1.20       0.20       2.50
    '''
    charge = 0
    spin  = 0
    basis_set = '3-21g'
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10


    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)
    
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    print(" Ecore: %12.8f" %ecore)
    


    H = Hamiltonian()
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = g
    H.ecore = ecore

    na = 2
    nb = 1
    

    ci = ci_solver()
    ci.max_iter  = 300
    ci.algorithm = "davidson"
    ci.thresh    = 1e-12 
    ci.init(H,na,nb,1)
    print(ci)
    ci.run()
    e1 = ci.results_e[0]
    v1 = ci.results_v
    #ci.results_v.shape = vfci.shape
    for i,ei in enumerate(ci.results_e):
        print(" State %5i: E: %12.8f Total E: %12.8f" %(i, ei, ei+ecore))

    ci = ci_solver()
    ci.max_iter  = 300
    ci.algorithm = "direct"
    ci.thresh    = 1e-12 
    ci.init(H,na,nb,1)
    print(ci)
    ci.run()
    e2 = ci.results_e[0]
    v2 = ci.results_v
    #ci.results_v.shape = vfci.shape
    for i,ei in enumerate(ci.results_e):
        print(" State %5i: E: %12.8f Total E: %12.8f" %(i, ei, ei+ecore))

    print(" Davidson: %18.14f" %(e1))
    print(" Direct  : %18.14f" %(e2))
    print(" Overlap : ", v1.T@v2)

    assert(abs(e1-e2)<1e-13)
    assert(abs(abs(v1.T@v2)-1)<1e-12)
    
if __name__== "__main__":
    test_1() 
