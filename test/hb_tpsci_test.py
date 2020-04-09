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


    do_fci = 1
    do_hci = 1
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,cas_nel,ecore=ecore,select_cutoff=1e-4,ci_cutoff=1e-4)
    if do_tci:
        
        clusters, clustered_ham, ci_vector = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 1 )
        
        
        
        
        edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
        ci_vector_ref = ci_vector.copy()
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector_ref.copy(), clustered_ham, 
                selection='heatbath', 
                thresh_cipsi=1e-6, 
                thresh_ci_clip=1e-7, 
                max_tucker_iter=3)
        etci += ecore





    print(" TCI:        %12.9f Dim:%6d"%(etci,len(ci_vector)))
    print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
    print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
    assert(abs(etci --108.855743070)< 1e-7)
    assert(len(ci_vector) == 248)
    assert(abs(efci   --108.85574521)< 1e-7)

def test_1():
    run(nproc=None)

def test_2():
    run(nproc=1)

if __name__== "__main__":
    test_1() 
    #test_2() 
