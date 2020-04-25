import sys, os
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
from timeit import default_timer as timer

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()


def test_1():
# {{{
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

        
    #efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    #print(" FCI energy: %12.8f" %efci)

    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState(clusters)
    if orb_basis == 'boys':
        ci_vector.init(((3,2),(2,3)))
        ci_vector.add_fockspace(((2,3),(3,2)))
        ci_vector.add_fockspace(((1,4),(4,1)))
        ci_vector.add_fockspace(((4,1),(1,4)))
    elif orb_basis == 'scf':
        ci_vector.init(((4,4),(1,1)))
        ci_vector.init(((4,3),(1,2)))
        ci_vector.init(((3,4),(2,1)))
        #ci_vector.init(((3,3),(2,2)))

    #ci_vector.add_fockspace(((2,2),(3,3)))
    #ci_vector.add_fockspace(((3,3),(2,2)))

    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

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
    ci_vector.expand_each_fock_space()
    #ci_vector.add_single_excitonic_states()
    #ci_vector.print_configs()
    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    #print("init DPS %16.8f"%(edps+ecore))

    print(" Build Hamiltonian. Space = ", len(ci_vector), flush=True)
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)

    n_roots=1
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))


    ci_vector.zero()
    ci_vector.set_vector(v0)


    if 1:
        print(" Compute Matrix Vector Product:", flush=True)
        start = timer()
        pt_vector1 = matvec1(clustered_ham, ci_vector)
        pt_vector1.prune_empty_fock_spaces()
        print(" Length of pt_vector", len(pt_vector1)) 
        for f in pt_vector1.fblocks():
            print(f, len(pt_vector1[f]))
        stop = timer()
        print(" Time lapse: ",(stop-start))

    print("\n\n")
    print(" Compute Matrix Vector Product:", flush=True)
    start = timer()
    pt_vector2 = matvec1_parallel1(clustered_ham, ci_vector)
    pt_vector2.prune_empty_fock_spaces()
    print(" Length of pt_vector", len(pt_vector2)) 
    #pt_vector.print()
    for f in pt_vector2.fblocks():
        print(f, len(pt_vector2[f]))
    stop = timer()
    print(" Time lapse: ",(stop-start))
    

    print("\n\n")
    print(" Compute Matrix Vector Product:", flush=True)
    start = timer()
    pt_vector3 = matvec1_parallel2(clustered_ham, ci_vector)
    pt_vector3.prune_empty_fock_spaces()
    print(" Length of pt_vector", len(pt_vector3)) 
    #pt_vector.print()
    for f in pt_vector3.fblocks():
        print(f, len(pt_vector3[f]))
    stop = timer()
    print(" Time lapse: ",(stop-start))

    for f in pt_vector1.fblocks():
        for c in pt_vector1[f]:
            assert(abs(pt_vector1[f][c] - pt_vector2[f][c]) < 1e-8)
            assert(abs(pt_vector1[f][c] - pt_vector3[f][c]) < 1e-8)
# }}}



if __name__== "__main__":
    test_1()