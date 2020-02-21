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

for ri in range(0,18):
    ###     PYSCF INPUT
    r0 = 2.50 + 0.1 * ri
    molecule = '''
    F
    H   1   {} 
    '''.format(r0,r0)
    charge = 0
    spin  = 0
    basis_set = 'ccpvdz'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 1
    cas_nstop =  19
    loc_start = 1
    loc_stop = 6
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2],[3,4],[5]]
    init_fspace = ((2, 2), (1, 1),(1,1))

    blocks = [range(0,4),range(4,8),range(8,12),range(12,16),range(16,18)]
    init_fspace = ((0, 0), (2, 2),(1, 1),(1,1),(0,0))

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    C = pmol.C

    do_fci = 0
    do_hci = 1
    do_tci = 0

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=5e-4,ci_cutoff=5e-4)

    #cluster using hcore
    idx = e1_order(h,cut_off = 1e-2)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    from pyscf import molden
    molden.from_mo(pmol.mol, 'h8.molden', C)
    print(h)
    if do_tci:
        #ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        #    thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=20,max_cipsi_iter=20)
        #ci_vector.print_configs()
        #tci_dim = len(ci_vector)
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
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-3, thresh_ci_clip=0, max_tucker_iter=3,asci_clip=0)
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-4, thresh_ci_clip=5e-2, max_tucker_iter=3,asci_clip=0)
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-5, thresh_ci_clip=1e-3, max_tucker_iter=3,asci_clip=0)
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-6, thresh_ci_clip=5e-4, max_tucker_iter=3,asci_clip=0)
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-7, thresh_ci_clip=1e-4, max_tucker_iter=3,asci_clip=0)

        #ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
        #    thresh_cipsi=1e-4, thresh_ci_clip=5e-3, max_tucker_iter=3,asci_clip=.1)
        print("init DPS",(edps+ecore))
        print("")
        print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
        print(" TPSCI(2):       %12.8f      Dim:%6d" % (etci2+ecore,len(pt_vector)))
        print("coefficient of dominant determinant")
        ci_vector.print_configs()
        tci_dim = len(ci_vector)
        etci = etci+ecore
        etci2 = etci2+ecore

    print(r0)
    print("  rad      FCI          Dim          HCI       Dim          TPSCI      Dim       TPSCI(2)")
    print(" %4.2f  %12.9f   %6d     %12.9f  %6d %12.9f %6d %12.9f"%(r0,efci,fci_dim,ehci,hci_dim,etci,tci_dim,etci2))
    exit()
