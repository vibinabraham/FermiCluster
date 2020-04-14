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


def test_save():
    
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
    
   
    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 4,  cmf_maxiter = 20 )
    

    ci_vector, pt_vector, etci, etci2, conv = bc_cipsi_tucker(ci_vector, clustered_ham, 
                                                        thresh_cipsi    = 1e-4, 
                                                        thresh_ci_clip  = 1e-7, 
                                                        max_tucker_iter = 2)
  


    filename = open('hamiltonian_file_test', 'wb')
    pickle.dump(clustered_ham, filename)
    filename.close()
    
    filename = open('state_file_test', 'wb')
    pickle.dump(ci_vector, filename)
    filename.close()

    np.save('ints_h.npy', h)
    np.save('ints_g.npy', g)
    print(" Computed energy : %16.14f " %etci)
    


def test_load():
    filename = open('hamiltonian_file_test', 'rb')
    clustered_ham = pickle.load(filename)
    filename.close()
    
    filename = open('state_file_test', 'rb')
    ci_vector = pickle.load(filename)
    filename.close()

    h = np.load('ints_h.npy')
    g = np.load('ints_g.npy')
    
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    v = ci_vector.get_vector()
    e = v.T @ H @ v

    print(" Computed energy : %16.14f " %e)
    
    assert(abs(e - -4.50903278169150) < 1e-12)


if __name__== "__main__":
    test_save() 
    test_load() 
