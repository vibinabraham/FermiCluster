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

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()




def run(nproc=None):
    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000

    from pyscf import gto, scf, mcscf, ao2mo

    r0 = 1.5

    molecule= '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)

    charge = 0
    spin  = 0
    basis_set = '6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'boys'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    #init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1,2,3],[4,5],[6,7]]
    init_fspace = ((3, 3), (1, 1), (1, 1))



    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    #cluster using hcore
    idx = e1_order(h,cut_off = 1e-4)
    h,g = reorder_integrals(idx,h,g)

    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    # intial state
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    ci_vector.print()

    # do cmf
    do_cmf = 1
    if do_cmf:
        # Get CMF reference
        e_cmf, cmf_conv, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter=10)
        #e_cmf, cmf_conv, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter=10, dm_guess=(dm_aa, dm_bb), diis=True)

    print(" Final CMF Total Energy %12.8f" %(e_cmf + ecore))

    # build cluster basis and operator matrices using CMF optimized density matrices
    for ci_idx, ci in enumerate(clusters):
        print(ci)
        fspaces_i = init_fspace[ci_idx]
        delta_e = 3
        fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_e) )

        print()
        print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
        ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=100, rdm1_a=rdm_a, rdm1_b=rdm_b)

        print(" Build mats for cluster ",ci.idx)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)


    hamiltonian_file = open('cmf_hamiltonian_file', 'wb')
    pickle.dump(clustered_ham, hamiltonian_file)


    ci_vector, pt_vector, etci, etci2, conv = bc_cipsi_tucker(ci_vector, clustered_ham, 
                                                        thresh_cipsi    = 1e-5, 
                                                        thresh_ci_clip  = 1e-6, 
                                                        max_tucker_iter = 20,
                                                        nproc=nproc)
    
    
    tci_dim = len(ci_vector)

    etci += ecore
    etci2 += ecore


    print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
    print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
    print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
    assert(abs(etci --108.855580011)< 1e-7)
    assert(tci_dim == 43)
    assert(abs(efci   --108.85574521)< 1e-7)

def test_1():
    run(nproc=None)

def test_2():
    run(nproc=1)

if __name__== "__main__":
    test_1() 
    test_2() 
