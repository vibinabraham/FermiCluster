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
    cas_nstart = 0
    cas_nstop = 10
    cas_nel = 14

    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1,2,7],[3,6],[4,5]]
   
    blocks = [[0,1],[2,3,4,5],[6,7,8,9]]

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
    ci_vector.init(((2,2),(4,4),(1,1)))

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

    ci_vector.expand_to_full_space()
    ci_vector.expand_each_fock_space()
    #ci_vector.add_single_excitonic_states()
    #ci_vector.print_configs()
    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    #print("init DPS %16.8f"%(edps+ecore))

    print(" Length of ci vector: ", len(ci_vector))
    precompute_cluster_basis_energies(clustered_ham)
    Hd_vector1 = ClusteredState(ci_vector.clusters)
    start = timer()
    Hd1 = update_hamiltonian_diagonal(clustered_ham, ci_vector.copy(), Hd_vector1)
    stop = timer()
    print(" Denomiator Time lapse: ",(stop-start))
    

    print(" Length of ci vector: ", len(ci_vector))
    start = timer()
    #Hd2 = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=1)
    Hd2 = build_hamiltonian_diagonal_parallel1(clustered_ham, ci_vector.copy())
    stop = timer()
    print(" Denomiator Time lapse: ",(stop-start))

    #assert(len(Hd_vector2) == len(Hd_vector1))
    for a in range(len(Hd_vector1)):
        assert(abs(Hd1[a]-Hd2[a]) < 1e-8)


if __name__== "__main__":
    test_1() 
