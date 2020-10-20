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
from pyscf_helper import *
import pyscf
ttt = time.time()
pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)

def test_1():
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    charge = 1
    spin  = 1
    basis_set = '3-21g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    blocks = [[0],[1,2,3],[4,5],[6,7]]
    init_fspace = ((1, 1),(1, 0), (0, 0), (0, 0))
    
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


    from pyscf import fci
    #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 200 
    cisolver.conv_tol = 1e-14 
    efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
    fci_dim = ci.shape[0]*ci.shape[1]
    d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
    print(" PYSCF 1RDM: ")
    occs = np.linalg.eig(d1)[0]
    [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
    with np.printoptions(precision=6, suppress=True):
        print(d1)
    print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    
    
    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, 
                        cmf_maxiter = 0)

    
    e0 = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)[0,0]
    # rotate the cluster states randomly
    np.random.seed(2)
    for ci in clusters:
        rotations = {}
        for fock in ci.basis:
            dim = ci.basis[fock].shape[1]
            G = np.random.random((dim,dim))
            U = scipy.linalg.expm(G-G.T)
            rotations[fock] = U
        
        ci.rotate_basis(rotations)

    
    e1 = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)[0,0]
    print(e0)
    print(e1)
    assert(e1>e0)
   

    print(" Build exact eigenstate")
    ci_vector.expand_to_full_space(clusters)
   
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)
    
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    vguess = ci_vector.get_vector()
    if H.shape[0] > 100 and abs(np.sum(vguess)) >0:
        e,v = scipy.sparse.linalg.eigsh(H,n_roots,v0=vguess,which='SA')
    else:
        e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))
    
    ci_vector.zero()
    ci_vector.set_vector(v0)
    
    tci_dim = len(ci_vector)

    rdm_a, rdm_b = build_1rdm(ci_vector, clusters)

    with np.printoptions(precision=6, suppress=True):
        print(" Difference from PYSCF density matrix")
        print(rdm_a + rdm_b - d1)
    assert(np.allclose(rdm_a + rdm_b, d1, atol=1e-4))
    
if __name__== "__main__":
    test_1() 
