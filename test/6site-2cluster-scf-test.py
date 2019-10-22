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

from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *
from bc_cipsi import *
import pyscf
ttt = time.time()

def run(beta, U, pbc=True): 
# {{{
    n_orb = 6 

    h, g = get_hubbard_params(n_orb,beta,U,pbc=True)
    np.random.seed(2)
    #tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
    #h += tmp + tmp.T

    if 1:
        Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

    do_fci = 1
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
        efci, ci = cisolver.kernel(h, g, h.shape[1], mol.nelectron, ecore=0)
        print(" FCI:        %12.8f"%efci)

    blocks = [[0,1,2],[3,4,5]]
    n_blocks = len(blocks)
    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState(clusters)
    ci_vector.init(((3,3),(0,0)))

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

    ci_vector_ref = ci_vector.copy()
    ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector_ref.copy(), clustered_ham, thresh_cipsi=1e-12, thresh_ci_clip=1e-12)
    print(" CIPSI: E0 = %12.8f E2 = %12.8f CI_DIM: %i" %(e0, e2, len(ci_vector)))
    if abs(e0 - efci)<1e-7:
        return True
    else:
        return False
# }}}

def test_1():
    assert(run(1,10))

def test_2():
    assert(run(1,1.))

def test_3():
    assert(run(1,.1))

def test_4():
    assert(run(1,10,pbc=False))

def test_5():
    assert(run(1,1.,pbc=False))

def test_6():
    assert(run(1,.1,pbc=False))

if __name__=="__main__":
    run(1,5)
