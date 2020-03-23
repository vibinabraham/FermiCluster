import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

for ri in range(0,26):
    ###     PYSCF INPUT
    r0 = 0.95 + 0.05 * ri
    molecule = '''
    N   {1}  {1}   {1} 
    H   {0}   {0}   0
    H   0   {0}   {0}
    H   {0}   0   {0}
    '''.format(r0,r0/2)
    r0 = np.sqrt(r0*r0*3/4)
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'ibmo'
    cas = True
    cas_nstart = 1
    cas_nstop =  8
    loc_start = 1
    loc_stop = 8
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    do_fci = 1
    do_hci = 1
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

    #cluster using hcore
    idx = e1_order(h,cut_off = 2e-1)
    h,g = reorder_integrals(idx,h,g)
    if do_tci:
        n_blocks = len(blocks)

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
        #ci_vector.add_single_excitonic_states()
        #ci_vector.print_configs()

        edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
        print(edps)

        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,selection='cipsi',
            thresh_cipsi=1e-6, thresh_ci_clip=1e-6, max_tucker_iter=4)
        print(" DPS     :       %12.8f      Dim:%6d" % (edps+ecore,1))
        etci = etci+ecore
        etci2 = etci2+ecore

        #ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        #    thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=4,max_cipsi_iter=10)
        tci_dim = len(ci_vector)
        ci_vector.clip(.005)
        ci_vector.print_configs()
        #ci_vector.normalize()
        #ci_vector.print_configs()
        #print(edps+ecore)

    for i in range(0,h.shape[0]):
        for j in range(0,h.shape[0]):
            print("%8.4f"%h[i,j],end='')
        print()

    print(r0)

    print("  rad      FCI          Dim          HCI       Dim          TPSCI      Dim       TPSCI(2)")
    print(" %4.2f  %12.9f   %6d     %12.9f  %6d %12.9f %6d %12.9f"%(r0,efci,fci_dim,ehci,hci_dim,etci,tci_dim,etci2))
    exit()
