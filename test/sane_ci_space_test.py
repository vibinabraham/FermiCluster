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

np.set_printoptions(suppress=True, precision=3, linewidth=1500)

def test_1():
    n_orb = 4
    U = 1.
    beta = 1.0

    h, g = get_hubbard_params(n_orb,beta,U,pbc=True)
    np.random.seed(2)
    #tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
    #h += tmp + tmp.T

    if 1:
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
    blocks = [[0,1],[2,3,4,5],[6,7]]
    blocks = [[0,1,6,7],[2,3,4,5]]
    blocks = [[0,1,2,3],[4,5],[6,7]]
    blocks = [[0,1,2,3,4,5],[6,7,8,9,10,11]]
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    blocks = [[0,1,2,3],[4,5],[6,7]]
    blocks = [[0,1,2,3],[4,5,6,7]]
    blocks = [[0],[1,2,3,4],[5]]
    blocks = [[0],[1,2,3,4],[5]]
    blocks = [[0,1,2,3]]
    #blocks = [[0,1],[2,3]]
    n_blocks = len(blocks)
    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState()
    ci_vector.init(clusters,((2,2),))
    ci_vector.add_fockspace(((2,1),))
    ci_vector.add_fockspace(((1,2),))
    ci_vector.add_fockspace(((3,1),))
    ci_vector.add_fockspace(((1,3),))

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
        print()
        print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
        fspaces_i = ci.possible_fockspaces()
        ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=1000)
        
        print(" Build operator matrices for cluster ",ci.idx)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)


    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()

    HH = build_full_hamiltonian(clustered_ham, ci_vector)
    print(HH)
    l,C = scipy.sparse.linalg.eigsh(HH,HH.shape[0],which='SA')
    sort_ind = np.argsort(l)
    l = l[sort_ind]
    C = C[:,sort_ind]
    print(l[0])

    for i in range(HH.shape[0]):
        print("%16.12f"%HH[i,i])

    assert(abs(HH[1,1] - HH[2,2])< 1e-12)
    assert(abs(HH[3,3] - HH[4,4])< 1e-12)


if __name__== "__main__":
    test_1() 
