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

def test_truncate_basis():
    # {{{
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
    H      2.00       1.00       0.00
    H      3.00       1.00       0.00
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
    blocks = [[0,1,2,3],[4,5],[6,7]]
    init_fspace = ((1, 1), (1, 0), (0, 1))
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((1, 1), (1, 1))
    blocks = [[0,2,4,6],[1,3,5,7]]
    init_fspace = ((2, 2), (0, 0))
    blocks = [[0,1,2,3,4,5,6,7]]
    init_fspace = ((2, 2),)
    
    
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

    do_fci = 1

    if do_fci:
        #efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore, max_cycle=200, conv_tol=12)
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
    

    
    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 20 ,
            cmf_thresh=1e-12, max_roots=1, delta_elec=1)
  
    if 1:
        ca,cb = clusters[0].split(init_fspace[0],4,h,g,thresh_schmidt=.0001)
        
        print(" Basis for ca")
        for f in ca.basis:
            print(f)
            print(ca.basis[f])
        print(" Basis for cb")
        for f in cb.basis:
            print(f)
            print(cb.basis[f])

        new_clusters = [ca,cb]
        new_clusters.extend(clusters[1:])
        clusters = new_clusters
        for ci in range(len(clusters)):
            clusters[ci].idx = ci
        
        [print(i) for i in clusters]
        clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
        print(" Add 1-body terms")
        clustered_ham.add_local_terms()
        clustered_ham.add_1b_terms(h)
        print(" Add 2-body terms")
        clustered_ham.add_2b_terms(g)
        
        clustered_ham.h = h
        clustered_ham.g = g
        ci_vector = ClusteredState()
        ci_vector.init(clusters, ((1,1),(1,1)))
        for ci_idx, ci in enumerate(clusters):
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)
   
    #ci_vector.expand_to_random_space(clusters, thresh=.2)
    ci_vector.expand_to_full_space(clusters)
    print(len(ci_vector))
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)
    e,v = np.linalg.eigh(H)
    e = e[0]
    v = v[:,0]
        
    ci_vector.zero()
    ci_vector.set_vector(v)
    ci_vector.print()
    
    print(" TPSCI:        %12.8f Dim:%6d"%(e+ecore,len(ci_vector)))
    print(" FCI  :        %12.8f Dim:%6d"%(efci,fci_dim))
    assert(np.isclose(e+ecore,efci,atol=1e-10))
    # }}}


if __name__== "__main__":
    test_truncate_basis() 
