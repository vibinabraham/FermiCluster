import sys, os
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
from timeit import default_timer as timer

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()


def test_1():

    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000

    from pyscf import gto, scf, mcscf, ao2mo

    r0 = 2.0

    molecule= '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)

    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'boys'
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1,2,7],[3,6],[4,5]]
   
    if orb_basis == 'boys':
        blocks = [[0,2,3,4],[1,5,6,7]]
    elif orb_basis == 'scf':
        blocks = [[0,1,2,3],[4,5,6,7]]

    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore



    #cluster using hcore
    #idx = e1_order(h,cut_off = 1e-4)
    #h,g = reorder_integrals(idx,h,g)

        
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    print(" FCI energy: %12.8f" %efci)

    init_fspace = ((4,4),(1,1))
    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace,
                                                        max_roots   = 100,
                                                        cmf_maxiter = 0
                                                        )

    ci_vector.add_fockspace(((4,3),(1,2)))
    ci_vector.add_fockspace(((3,4),(2,1)))
    


    #ci_vector.add_fockspace(((2,2),(3,3)))
    #ci_vector.add_fockspace(((3,3),(2,2)))


    #ci_vector.expand_to_full_space()
    ci_vector.expand_each_fock_space(clusters)
    #ci_vector.add_single_excitonic_states()
    #ci_vector.print_configs()
    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    #print("init DPS %16.8f"%(edps+ecore))

    print(" Build Hamiltonian. Space = ", len(ci_vector), flush=True)
    #H = build_full_hamiltonian_open(clustered_ham, ci_vector)
    start = timer()
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)
    stop = timer()
    print(" Time lapse: ",(stop-start))
    n_roots=1
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    e1 = 1*e0
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))

    
    print(" Build Hamiltonian. Space = ", len(ci_vector), flush=True)
    start = timer()
    ci_vector.randomize_vector()
    ci_vector_small = ci_vector.copy()
    ci_vector_small.clip(.1)
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector_small)
    H = grow_hamiltonian_parallel(H, clustered_ham, ci_vector, ci_vector_small)
    stop = timer()
    print(" Time lapse: ",(stop-start))
    n_roots=1
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    e2 = 1*e0
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))

    print(" Now do the serial version")
    start = timer()
    H = build_full_hamiltonian(clustered_ham, ci_vector)
    stop = timer()
    print(" Time lapse: ",(stop-start))
    
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    e3 = 1*e0
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))

    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    assert(abs(e2-e1) < 1e-8)
    assert(abs(e3-e1) < 1e-8)








if __name__== "__main__":
    test_1() 
    #test_2() 
