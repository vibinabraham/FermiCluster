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
pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)

def test_1():
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    charge = 0
    spin  = 0
    basis_set = '3-21g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (0, 0))
    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == nelec)
        nelec = cas_nel


    #Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    #cluster using hcore
    #idx = e1_order(h,cut_off = 1e-2)
    #h,g = reorder_integrals(idx,h,g)

    do_fci = 1
    do_hci = 1
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,4,ecore=ecore)#,select_cutoff=2e-3,ci_cutoff=2e-3)
    if do_tci:
        
        clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 0 )

        ci_vector, pt_vector, etci, etci2, conv = tpsci_tucker(ci_vector, clustered_ham, 
                                                            thresh_cipsi    = 1e-4, 
                                                            thresh_ci_clip  = 1e-7, 
                                                            matvec          = 1,
                                                            max_tucker_iter = 2)
        
        tci_dim = len(ci_vector)

    print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
    print(" HCI:        %12.9f Dim:%6d"%(ehci-ecore,hci_dim))
    print(" FCI:        %12.9f Dim:%6d"%(efci-ecore,fci_dim))
    assert(abs(etci --3.08117519) < 1e-8)
    assert(abs(etci2 --3.08150334) < 1e-8)
    assert(tci_dim == 9)
    assert(abs(efci -ecore --3.08154574) < 1e-7)
    
if __name__== "__main__":
    test_1() 
