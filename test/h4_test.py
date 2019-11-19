import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *
from bc_cipsi import *
import pyscf
ttt = time.time()
pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues

def test_1():
    ttt = time.time()

    n_cluster_states = 1000

    from pyscf import gto, scf, mcscf, ao2mo
    mol = gto.Mole()
    mol.atom = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    mol.charge      = +0
    mol.spin        = +0
    mol.max_memory  = 1000 # MB

    mol.basis = '3-21g'

    myhf = scf.RHF(mol).run()
    print(myhf.mo_energy)
    #exit()
    n_orb = myhf.mo_coeff.shape[1]
    #g.shape = (n_orb,n_orb,n_orb,n_orb)
    enu = myhf.energy_nuc()
    S = mol.intor('int1e_ovlp_sph')

    local = 'lowdin'
    local = 'p'

    if local == 'lowdin':
        print("Using lowdin orthogonalized orbitals")
        #forming S^-1/2 to transform to A and B block.
        sal, svec = np.linalg.eigh(S)
        idx = sal.argsort()[::-1]
        sal = sal[idx]
        svec = svec[:, idx]
        sal = sal**-0.5
        sal = np.diagflat(sal)
        X = svec @ sal @ svec.T
        C = cp.deepcopy(X)

        h = C.T.dot(myhf.get_hcore()).dot(C)
        g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))

    else:
        h = myhf.mo_coeff.T.dot(myhf.get_hcore()).dot(myhf.mo_coeff)
        g = ao2mo.kernel(mol,myhf.mo_coeff,aosym='s4',compact=False).reshape(4*((n_orb),))
    print(g.shape)

    do_fci = 1
    do_hci = 1

    if do_fci:
        # FCI
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI(mol)
        efci, ci = cisolver.kernel(h, g, h.shape[1], mol.nelectron, ecore=0)
        fci_dim = ci.shape[0]*ci.shape[1]
        print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
        print("FCI %10.8f"%(efci+enu))

    if do_hci:
        #heat bath ci
        from pyscf import mcscf
        from pyscf.hci import hci
        cisolver = hci.SCI(mol)
        ehci, civec = cisolver.kernel(h, g, h.shape[1], mol.nelectron, ecore=0,verbose=4)
        hci_dim = civec[0].shape[0]
        print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
        print("HCI %10.8f"%(ehci+enu))

    blocks = [[0,1,2,3],[4,5,6,7]]
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    n_blocks = len(blocks)
    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState(clusters)
    ci_vector.init(((2,2),(0,0)))
    #ci_vector.init(((1,1),(1,1),(0,0),(0,0)))

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
        ci.form_eigbasis_from_local_operator(opi,max_roots=n_cluster_states)


    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx)
        c.build_op_matrices()

    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()

    thresh_conv = 1e-8
    #ci_vector, pt_vector, e0, e2 = bc_cipsi_tucker(ci_vector.copy(),clustered_ham, thresh_cipsi=1e-5, thresh_ci_clip=1e-5,hshift=1e-8, max_tucker_iter=8)
    ci_vector, pt_vector, e0, e2 = bc_cipsi_tucker(ci_vector.copy(),clustered_ham, thresh_cipsi=1e-6, thresh_ci_clip=1e-6,hshift=1e-8, max_tucker_iter=8)
    print(" CIPSI: E0 = %12.8f E2 = %12.8f CI_DIM: %i" %(e0, e2, len(ci_vector)))
    print(e2+enu)
    bcci_dim = len(ci_vector)

    print("BCCI:        %12.8f Dim:%6d"%(e0,bcci_dim))
    print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
    print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    assert(abs(e0 --3.08152531) < 1e-7)
    assert(abs(e2 --3.08154325) < 1e-7)
    assert(abs(bcci_dim - 48)<1e-15)
    assert(abs(efci --3.08154574) < 1e-7)
    
if __name__== "__main__":
    test_1() 
