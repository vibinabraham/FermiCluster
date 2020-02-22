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

def test_1():
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
    #basis_set = 'sto-3g'
    basis_set = '6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    #blocks = [[i] for i in range(4)] 
    #init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0))
    blocks = [[i] for i in range(8)] 
    init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))
   
    #blocks = [[0],[1,2],[3]]
    #init_fspace = ((1, 1), (1, 1), (0, 0))
    
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

    h_save = cp.deepcopy(h)
    g_save = cp.deepcopy(g)
    
    print(" Ecore: %12.8f" %ecore)
    
    #cluster using hcore
    #idx = e1_order(h,cut_off = 1e-2)
    #h,g = reorder_integrals(idx,h,g)

    do_fci = 0

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
    

    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
   

    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.combine_common_terms(iprint=1)
   
    
    # Get CMF reference
    cmf(clustered_ham, ci_vector, h, g, max_iter=10)

    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g)

    

if __name__== "__main__":
    test_1() 
