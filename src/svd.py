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
    H      1.00       0.00       0.00
    H      0.00       0.10       2.00
    H      1.00       0.10       2.00
    H      0.00       0.20       4.00
    H      1.00       0.20       4.00
    '''
    charge = 0
    spin  = 0
    basis_set = '3-21g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    init_fspace = ((1, 1), (1, 1), (1, 1))
    
    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == nelec)
        nelec = cas_nel


    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)
    
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    print(" Ecore: %12.8f" %ecore)
    


    if 0:
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 300 
        cisolver.conv_tol = 1e-14 
        efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
        #efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
        print(" E(FCI): %12.8f" %efci)
    
    
    H = Hamiltonian()
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = g
    H.ecore = ecore

    clusters, clustered_ham, ci_vector, cmf_out  = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 100,  cmf_maxiter = 0 )
    rdm_a, rdm_b = build_1rdm(ci_vector, clusters)

    #for ci in clusters:
    #    ci.form_dmet_basis(h,g,rdm_a,rdm_b, thresh=.0001, do_embedding=False)
    #    #ci.form_dmet_basis(h,g,rdm_a,rdm_b, thresh=.0001, do_embedding=True)

    na = 2
    nb = 4

    ci = ci_solver()
    ci.max_iter = 300
    ci.algorithm = "davidson2"
    ci.init(H,na,nb,1)
    print(ci)
    ci.run(iprint=1)
    for i,ei in enumerate(ci.results_e):
        print(" State %5i: E: %12.8f Total E: %12.8f" %(i, ei, ei+ecore))
    ci.svd_state(4,8, thresh=.001)



def test_2():
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
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((1, 1), (1, 1))
    
    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == nelec)
        nelec = cas_nel


    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)
    
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    print(" Ecore: %12.8f" %ecore)
    


    if 0:
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 300 
        cisolver.conv_tol = 1e-14 
        efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
        #efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
        print(" E(FCI): %12.8f" %efci)
         
    
    H = Hamiltonian()
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = g
    H.ecore = ecore

    clusters, clustered_ham, ci_vector, cmf_out  = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 100,  cmf_maxiter = 0 )
    rdm_a, rdm_b = build_1rdm(ci_vector, clusters)

    #for ci in clusters:
    #    ci.form_dmet_basis(h,g,rdm_a,rdm_b, thresh=.0001, do_embedding=False)
    #    #ci.form_dmet_basis(h,g,rdm_a,rdm_b, thresh=.0001, do_embedding=True)

    na = 2
    nb = 2
    
    if 0:
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 300 
        cisolver.conv_tol = 1e-14 
        efci, vfci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
        #efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
        print(" E(FCI): %12.8f" %efci)
        print(vfci)
        for i in range(vfci.shape[0]):
            print(pyscf.fci.cistring.addr2str(h.shape[0], nelec[0], i))
        exit()

    ci = ci_solver()
    ci.max_iter  = 300
    ci.algorithm = "direct"
    ci.algorithm = "davidson2"
    ci.thresh    = 1e-12 
    ci.init(H,na,nb,1)
    print(ci)
    ci.run()
    #ci.results_v.shape = vfci.shape
    for i,ei in enumerate(ci.results_e):
        print(" State %5i: E: %12.8f Total E: %12.8f" %(i, ei, ei+ecore))
    ci.svd_state(4,4, thresh=.001)
    exit()

if __name__== "__main__":
    test_1() 
    #test_2() 
