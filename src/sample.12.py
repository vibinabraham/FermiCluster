import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *
from bc_cipsi import *
import pyscf
ttt = time.time()

if __name__=="__main__":
    #client = Client(processes=True)
    #client = Client(processes=False)
    n_orb =  12 
    U = 5.
    beta = 1.0
    
    h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
    np.random.seed(2)
    #tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
    #h += tmp + tmp.T
    
    if 0:
        Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)
    
    do_fci = 0
    if do_fci:
        # FCI
        from pyscf import gto, scf, ao2mo, fci, cc
        pyscf.lib.num_threads(1)
        mol = gto.M(verbose=3)
        mol.nelectron = n_orb
        # Setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
        # attribute) to be used in the post-HF calculations.  Without this parameter,
        # some post-HF method (particularly in the MO integral transformation) may
        # ignore the customized Hamiltonian if memory is not enough.
        mol.incore_anyway = True
        cisolver = fci.direct_spin1.FCI(mol)
        #e, ci = cisolver.kernel(h1, eri, h1.shape[1], 2, ecore=mol.energy_nuc())
        e, ci = cisolver.kernel(h, g, h.shape[1], mol.nelectron, ecore=0)
        print(" FCI:        %12.8f"%e)
    
    blocks = [[0,5],[1,4],[2,3]]
    blocks = [[0],[1],[2],[3],[4],[5]]
    blocks = [[0],[1],[2],[3],[4],[5],[6],[7]]
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    blocks = [[0,1],[2,3]]
    blocks = [[0,1],[2,3,4,5],[6,7]]
    blocks = [[0,1,6,7],[2,3,4,5]]
    blocks = [[0,1,2,3],[4,5],[6,7]]
    blocks = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    blocks = [[0,1,2],[3,4,5]]
    blocks = [[0,1,2,3],[4,5],[6,7]]
    blocks = [[0,1,2,3],[4,5,6,7]]
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    n_blocks = len(blocks)
    clusters = []
    
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector = ClusteredState(clusters)
    #ci_vector.init(((2,2),(0,0)))
    #ci_vector.init(((2,2),(2,2),(0,0)))
    #ci_vector.init(((2,2),(2,2),(0,0),(0,0)))
    #ci_vector.init(((2,2),(2,2)))
    #ci_vector.init(((1,1),(1,1),(1,1)))
    #ci_vector.init(((2,1),(1,2)))
    #ci_vector.init(((1,1),(1,1),(1,1),(0,0),(0,0),(0,0)))
    #ci_vector.init(((1,1),(1,1),(1,1),(1,1)))
    #ci_vector.init(((2,2),(2,2),(0,0),(0,0)))
    #ci_vector.init(((1,1),(1,1)))
    #ci_vector.init(((2,2),(2,2),(0,0)))
    #ci_vector.init(((3,3),(0,0)))
    #ci_vector.init(((4,4),(0,0)))
    #ci_vector.init(((4,4),(0,0),(0,0)))
    ci_vector.init(((2,2),(2,2),(2,2)))
    #ci_vector.init(((1,1),(1,1),(1,1),(1,1)))
    #ci_vector.init(((2,2),(1,1),(1,1)))
    #ci_vector.init(((3,3),(3,3)))
    #ci_vector.init(((4,4),(2,2),(0,0)))
    #ci_vector.init(((2,2),(2,2),(2,2)))
    #ci_vector.init(((4,4),(4,4),(4,4),(0,0),(0,0),(0,0)))
    #ci_vector.init(((1,1),(1,1),(1,1),(1,1),(0,0),(0,0),(0,0),(0,0)))
    #ci_vector.init(((2,2),(2,2),(2,2),(2,2),(2,2),(2,2)))
    
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.combine_common_terms(iprint=1)
    
    print(" Build cluster basis")
    for ci_idx, ci in enumerate(clusters):
        assert(ci_idx == ci.idx)
        print(" Extract local operator for cluster",ci.idx)
        opi = clustered_ham.extract_local_operator(ci_idx)
        print()
        print()
        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
        ci.form_eigbasis_from_local_operator(opi,max_roots=1000)
    
    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx)
        c.build_op_matrices()
    
    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()
    
    ci_vector, pt_vector, e0, e2 = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-5, thresh_ci_clip=1e-5)
    
