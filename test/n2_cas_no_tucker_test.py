import sys, os
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
    basis_set = 'ccpvdz'

    ###     TPSCI BASIS INPUT
    orb_basis = 'boys'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10

    ###     TPSCI CLUSTER INPUT
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

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,cas_nel,ecore=ecore,select_cutoff=1e-4,ci_cutoff=1e-4)
        
         
    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 0 )


    ci_vector, pt_vector, etci, etci2, conv = bc_cipsi_tucker(ci_vector, clustered_ham, 
                                                        thresh_cipsi    = 1e-5, 
                                                        thresh_ci_clip  = 1e-6, 
                                                        max_tucker_iter = 0)
    
    tci_dim = len(ci_vector)

    etci += ecore
    etci2 += ecore

    print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
    print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
    print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
    assert(abs(etci --108.756551406)< 1e-7)
    assert(abs(etci2 --108.756912357)< 1e-7)
    assert(abs(tci_dim - 106)<1e-15)
    assert(abs(efci   --108.756976573)< 1e-7)


if __name__== "__main__":
    test_1() 
