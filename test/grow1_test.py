import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import tools 

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)


def test1():
    
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      2.00       0.00       0.00
    H      3.00       0.00       0.00
    '''
    charge = 0
    spin  = 0
    basis_set = '3-21g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((1, 1), (1, 1))
    
    
    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == nelec)
        nelec = cas_nel


    #Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    n_orb = pmol.n_orb

    print(" Ecore: %12.8f" %ecore)
    

    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
   
    clusters, clustered_ham, ci_vector = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 4,  cmf_maxiter = 20 )
    

    ci_vector, pt_vector, etci, etci2, conv = bc_cipsi_tucker(ci_vector, clustered_ham, 
                                                        thresh_cipsi    = 1e-4, 
                                                        thresh_ci_clip  = 1e-7, 
                                                        max_tucker_iter = 2)
  
    l1 = len(ci_vector)
        #hamiltonian_file = open('hamiltonian_file', 'wb')
        #pickle.dump(clustered_ham, hamiltonian_file)


    #hamiltonian_file = open('hamiltonian_file', 'rb')
    #clustered_ham = pickle.load(hamiltonian_file)
    #clusters = clustered_ham.clusters
    
    #ci_vector = ClusteredState(clusters)
    #ci_vector.init(init_fspace)


    for ci in clusters:
        ci.grow_basis_by_energy()
        
        print(" Build operator matrices for cluster ",ci.idx)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)
    
    print(" Size of basis2: ",len(ci_vector))

    test1 = matvec1(clustered_ham, ci_vector)
    e1 = test1.dot(ci_vector)
    print(" Energy of old cipsi state in new larger basis %16.12f" %e1, len(ci_vector), l1)
    assert(abs(e1-etci) < 1e-12)
    
    ci_vector, pt_vector, etci, etci2, conv = bc_cipsi_tucker(ci_vector, clustered_ham, 
                                                        thresh_cipsi    = 1e-4, 
                                                        thresh_ci_clip  = 1e-7, 
                                                        max_tucker_iter = 2)
    l2 = len(ci_vector)
    
    print(" E(TPSCI-old)  = %12.8f" %e1)
    print(" E(TPSCI)      = %12.8f" %etci)
    print(" E(TPSCI2)     = %12.8f" %etci2)
    print(" E(FCI)        = %12.8f" %(efci-ecore))
    
    #assert(abs(efci --2.21837081) < 1e-7)
    assert(abs(etci --4.51129093) < 1e-7)

if __name__== "__main__":
    test1() 
