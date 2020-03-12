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
from numpy.linalg import norm


from ci_string import *


def test1():
# {{{
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
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10
    
    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    #init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1],[2,7],[3,6],[4,5]]
    init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    
    
    
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    print("Ecore:%16.8f"%ecore)
    C = pmol.C
    K = pmol.K
    mol = pmol.mol
    mo_energy = pmol.mf.mo_energy
    dm_aa = pmol.dm_aa
    dm_bb = pmol.dm_bb
    
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    print(" FCI: %12.8f (elec)" %(efci-ecore)) 
    print(" FCI: %12.8f (total)" %efci) 
    do_fci = 1
    do_hci = 1
    do_tci = 1
    
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
    #clustered_ham.add_1b_terms(cp.deepcopy(h))
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.add_2b_terms(cp.deepcopy(g))
    #clustered_ham.combine_common_terms(iprint=1)
    
    
    do_cmf = 1
    if do_cmf:
        # Get CMF reference
        #cmf(clustered_ham, ci_vector, cp.deepcopy(h), cp.deepcopy(g), max_iter=4)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=2)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=50,dm_guess=(dm_aa,dm_bb),diis=True)
        cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=2,dm_guess=(dm_aa,dm_bb),diis=True)
    else:
        print(" Build cluster basis and operators")
        for ci_idx, ci in enumerate(clusters):
            ci.form_eigbasis_from_ints(h,g,max_roots=1)
        
            print(" Build new operators for cluster ",ci.idx)
            ci.build_op_matrices()
  
    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print(" Energy of reference TPS: %12.8f (elec)"%(edps))
    print(" Energy of reference TPS: %12.8f (total)"%(edps+ecore))
  
    ci_vector.expand_to_full_space()
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e0.real,len(ci_vector)))


    print(" ---------------- Combine -------------------")
    c12 = join_bases(clusters[0], clusters[1]) 
    new_clusters = [c12]
    new_clusters.extend(clusters[2:])
    [print(i) for i in new_clusters]
    clusters = new_clusters 
    for ci in range(len(clusters)):
        clusters[ci].idx = ci
    init_fspace = ((3, 3), (1, 1), (1, 1))
    

    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    
    
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(cp.deepcopy(h))
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(cp.deepcopy(g))
    #clustered_ham.combine_common_terms(iprint=1)
    
    do_cmf = 0
    if do_cmf:
        # Get CMF reference
        cmf(clustered_ham, ci_vector, h, g, max_iter=10)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=2)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=5,dm_guess=(dm_aa,dm_bb),diis=True)

    print(" Build cluster operators")
    [ci.build_op_matrices() for ci in clusters]
    
    edps2 = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print(" Energy of reference TPS: %12.8f (elec)"%(edps2))
    print(" Energy of reference TPS: %12.8f (total)"%(edps2+ecore))
    
    ci_vector.expand_to_full_space()
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v1 = v[:,0]
    e1 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e1.real,len(ci_vector)))

    
    assert(abs(e1-e0) < 1e-8)
# }}}

def test2():
    """
    Test that the hilbert space spanned by a set of small clusters, 
    and a set of joined clusters is identical, even when truncated
    
    starts from a cmf+pt2 calculation, does a Tucker decomposition and truncates, 
    then does an exact diagonalization in that subspace.
    Then combines two clusters and then does another exact diagonalization.
    These two calucations should give the exact same results.
    """
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
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10
    
    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    #init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1],[2,7],[3,6],[4,5]]
    init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    
    
    
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    print("Ecore:%16.8f"%ecore)
    C = pmol.C
    K = pmol.K
    mol = pmol.mol
    mo_energy = pmol.mf.mo_energy
    dm_aa = pmol.dm_aa
    dm_bb = pmol.dm_bb
    
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    print(" FCI: %12.8f (elec)" %(efci-ecore)) 
    print(" FCI: %12.8f (total)" %efci) 
    do_fci = 1
    do_hci = 1
    do_tci = 1
    
    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector_ref = ClusteredState(clusters)
    ci_vector_ref.init(init_fspace)
    
    
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    #clustered_ham.add_1b_terms(cp.deepcopy(h))
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.add_2b_terms(cp.deepcopy(g))
    #clustered_ham.combine_common_terms(iprint=1)
    
    
    do_cmf = 0
    if do_cmf:
        # Get CMF reference
        #cmf(clustered_ham, ci_vector, cp.deepcopy(h), cp.deepcopy(g), max_iter=4)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=2)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=50,dm_guess=(dm_aa,dm_bb),diis=True)
        cmf(clustered_ham, ci_vector_ref, h, g, max_iter=50,max_nroots=40,dm_guess=(dm_aa,dm_bb),diis=True)
    else:
        print(" Build cluster basis and operators")
        for ci_idx, ci in enumerate(clusters):
            ci.form_eigbasis_from_ints(h,g)
            #ci.form_eigbasis_from_ints(h,g,max_roots=1)
        
            print(" Build new operators for cluster ",ci.idx)
            ci.build_op_matrices()
 
    ci_vector = ci_vector_ref.copy()
    e0 = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print(" Energy of reference TPS: %12.8f (elec)"%(e0))
    print(" Energy of reference TPS: %12.8f (total)"%(e0+ecore))
  
    
    e2, pt_vector = compute_pt2_correction(ci_vector, clustered_ham, e0, nproc=4)
    print(" Energy of TPS(2): %12.8f (elec)"%(e0+e2))
    print(" Energy of TPS(2): %12.8f (total)"%(e0+e2+ecore))

    ci_vector.add(pt_vector)
    ci_vector.normalize()
    
    hosvd(ci_vector, clustered_ham, trim=1e-3, hshift=None)

    ci_vector = ci_vector_ref.copy()
    ci_vector.expand_to_full_space()
    print(" Build Hamiltonian in the full space after HOSVD. Dimension:", len(ci_vector))
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    #H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e0.real,len(ci_vector)))
    

    clusters, clustered_ham = join(clusters,0,1,h,g)
    init_fspace = ((3, 3), (1, 1), (1, 1))

    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    
    ci_vector.expand_to_full_space()
    print(" Build Hamiltonian in the full space after HOSVD. Dimension:", len(ci_vector))
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    #H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = scipy.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v1 = v[:,0]
    e1 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e0.real,len(ci_vector)))
    
    assert(abs(e1-e0) < 1e-9)

if __name__ == "__main__":
    test1()
    test2()
