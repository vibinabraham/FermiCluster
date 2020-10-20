import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from fermicluster import *
from pyscf_helper import *
import pyscf
ttt = time.time()
pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)

def test_1():
    ttt = time.time()# {{{

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      0.00       0.10       2.50
    H      1.00       0.10       2.50
    H      0.00       0.20       4.50
    H      1.00       0.20       4.50
    '''
    charge = 0
    spin  = 0
    basis_set = '3-21g'
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    init_fspace = ((1, 1), (1, 1), (1, 1))
    
    blocks = [[0,1],[2,3],[4,5]]
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
    
    H = Hamiltonian()
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = g
    H.ecore = ecore
    
    if 1:
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 300 
        cisolver.conv_tol = 1e-14 
        efci, vfci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
        print(" E(FCI): %12.8f %12.8f" %(efci,efci-ecore))

    clusters, clustered_ham, ci_vector, cmf_out  = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 3,  cmf_maxiter = 0 )
    rdm_a, rdm_b = build_1rdm(ci_vector, clusters)
    
    if 1:
        for ci in clusters:
            ci.form_schmidt_basis(h,g,rdm_a,rdm_b, thresh_schmidt=1e-3, do_embedding=True)
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)
    
    ci_vector.expand_to_full_space(clusters)
    print(len(ci_vector))
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    n_roots=1
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    e1 = 1*e0
    print(" E(FCI):                             %12.8f %12.8f" %(efci,efci-ecore))
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))
    assert(abs(ecore+e0.real - -3.28442748) < 1e-7) 
# }}}

def test_2():
    """
    Test that the SVD exactly finds the minimal space for a 2 cluster problem,
    without any complications from environment embedding
    """
    # {{{
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      0.00       2.20       1.30
    H      1.00       2.00       1.30
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
    

    from pyscf import fci
    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 300 
    cisolver.conv_tol = 1e-14 
    efci, vfci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
    print(" E(FCI): %12.8f %12.8f" %(efci,efci-ecore))

    H = Hamiltonian()
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = g
    H.ecore = ecore

    clusters, clustered_ham, ci_vector, cmf_out  = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 100,  cmf_maxiter = 0 )
    rdm_a, rdm_b = build_1rdm(ci_vector, clusters)

    if 1:
        for ci in clusters:
            ci.form_schmidt_basis(h,g,rdm_a,rdm_b, thresh_schmidt=1e-3)
            #ci.form_schmidt_basis(h,g,rdm_a,rdm_b, thresh=.0001, do_embedding=True)
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)
    
    ci_vector.expand_to_full_space(clusters)
    print(len(ci_vector))
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    n_roots=1
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    e1 = 1*e0
    print(" E(FCI):                             %12.8f %12.8f" %(efci,efci-ecore))
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))
    assert(abs(ecore+e0.real - -2.23993069) < 1e-7) 
# }}}

def test_3():
    """
    Test the CMF inside of the subspace determined from schmidt truncation
    """
    # {{{
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      0.00       2.20       1.30
    H      1.00       2.00       1.30
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
    

    from pyscf import fci
    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 300 
    cisolver.conv_tol = 1e-14 
    efci, vfci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
    print(" E(FCI): %12.8f %12.8f" %(efci,efci-ecore))

    H = Hamiltonian()
    H.S = np.eye(h.shape[0])
    H.C = H.S
    H.t = h
    H.V = g
    H.ecore = ecore

    clusters, clustered_ham, ci_vector, cmf_out  = system_setup(h, g, ecore, blocks, init_fspace, max_roots = 100,  cmf_maxiter = 10 )
    rdm_a, rdm_b = build_1rdm(ci_vector, clusters)
    
    e0 = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)[0,0]
    e2aa, _ = compute_pt2_correction(ci_vector, clustered_ham, e0, pt_type='mp')
    e2bb, _ = compute_pt2_correction(ci_vector, clustered_ham, e0, pt_type='en')
    

    if 1:
        for ci in clusters:
            ci.form_schmidt_basis(h,g,rdm_a,rdm_b, thresh_schmidt=1e-3)
            #ci.form_schmidt_basis(h,g,rdm_a,rdm_b, thresh=.0001, do_embedding=True)
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)
            ci.build_effective_cmf_hamiltonian(h,g,rdm_a,rdm_b)
   
    e0 = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)[0,0]
    e2a, _ = compute_pt2_correction(ci_vector, clustered_ham, e0, pt_type='mp')
    e2b, _ = compute_pt2_correction(ci_vector, clustered_ham, e0, pt_type='en')
    
    
    if 1:
        for ci in clusters:
            print()
            print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci.idx)
            fspaces_i = ci.basis.keys()
            ci.form_fockspace_eigbasis(h, g, fspaces_i, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=ecore, subspace=True)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)
            ci.build_effective_cmf_hamiltonian(h,g,rdm_a,rdm_b)
    

    e0 = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)[0,0]
    e2c, _ = compute_pt2_correction(ci_vector, clustered_ham, e0, pt_type='mp')
    e2d, _ = compute_pt2_correction(ci_vector, clustered_ham, e0, pt_type='en')

    
    ci_vector.expand_to_full_space(clusters)
    print("length of ci_vector", len(ci_vector))
    ci_vector.print_configs()
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    n_roots=1
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.sparse.linalg.eigsh(H,n_roots,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    e1 = 1*e0
    print(" E(CMF):                             %12.8f" %(cmf_out[0]+ecore))
    print(" E(MP2)            :                 %12.8f  "%(ecore+cmf_out[0]+e2aa))
    print(" E(MP2) Schmidt    :                 %12.8f  "%(e2a+cmf_out[0]+ecore))
    print(" E(MP2) Canonical  :                 %12.8f  "%(ecore+cmf_out[0]+e2c))
    print(" E(EN)             :                 %12.8f  "%(ecore+cmf_out[0]+e2bb))
    print(" E(EN)  Schmidt    :                 %12.8f  "%(ecore+cmf_out[0]+e2b))
    print(" E(EN)  Canonical  :                 %12.8f  "%(ecore+cmf_out[0]+e2d))
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(ecore+e0.real,len(ci_vector)))
    print(" E(FCI):                             %12.8f %12.8f" %(efci,efci-ecore))
    #assert(abs(ecore+e2aa - 1.87008352) < 1e-7) 
    #assert(abs(ecore+e2bb - ) < 1e-7) 
# }}}

if __name__== "__main__":
    test_1() 
    test_2() 
    test_3() 
