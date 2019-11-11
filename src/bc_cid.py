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

def bc_cid(ci_vector, clustered_ham, thresh_cipsi=1e-4, thresh_conv=1e-8, n_roots=1):
# {{{
    print(" Compute diagonal elements",flush=True)
    # compute local states energies
    precompute_cluster_basis_energies(clustered_ham)
    print(" done.",flush=True)

    pt_vector = ci_vector.copy()
    Hd_vector = ClusteredState(ci_vector.clusters)

    print()
    print(" ===================================================================")
    print("             CID epsilon: %12.8f" %(thresh_cipsi))
    print(" ===================================================================")
    print(" Build full Hamiltonian",flush=True)
    H = build_full_hamiltonian(clustered_ham, ci_vector)

    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    vguess = ci_vector.get_vector()
    if H.shape[0] > 100 and abs(np.sum(vguess)) >0:
        e,v = scipy.sparse.linalg.eigsh(H,n_roots,v0=vguess,which='SA')
    else:
        e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))

    ci_vector.zero()
    ci_vector.set_vector(v0)

    ci_dim = len(ci_vector)
    return ci_vector, e0
# }}}

def bc_cid_tucker(ci_vector, clustered_ham, thresh_cipsi=1e-4, thresh_cipsi_conv=1e-8, 
    thresh_tucker_conv = 1e-6, max_tucker_iter=20, tucker_state_clip=None):
    """
    Run iterations of BR-CID to make the tucker decomposition self-consistent
    """
# {{{
    if tucker_state_clip == None:
        tucker_state_clip = thresh_cipsi/10.0
    e_prev = 0
    e_last = 0
    ci_vector_ref = ci_vector.copy()
    for brdm_iter in range(max_tucker_iter):
        ci_vector, e0 = bc_cid(ci_vector_ref.copy(), clustered_ham, thresh_cipsi=1e-14)
        print(" CID: E0 = %12.8f  CI_DIM: %i" %(e0, len(ci_vector)))
      
        if abs(e_prev-e0) < thresh_tucker_conv:
            print(" Converged BRDMs")
            break
        e_prev = e0
        print(" Reduce size of 1st order wavefunction")
        print(" Before:",len(ci_vector))
        ci_vector.clip(tucker_state_clip)
        ci_vector.normalize()
        print(" After:",len(ci_vector))
        for ci in clustered_ham.clusters:
            print()
            print(" Compute BRDM",flush=True)
            rdms = build_brdm(ci_vector, ci.idx)
            print(" done.",flush=True)
            #rdms = build_brdm(ci_vector, ci.idx)
            norm = 0
            rotations = {}
            for fspace,rdm in rdms.items():
                print(" Diagonalize RDM for Cluster %2i in Fock space:"%ci.idx, fspace,flush=True)
                n,U = np.linalg.eigh(rdm)
                idx = n.argsort()[::-1]
                n = n[idx]
                U = U[:,idx]
                norm += sum(n)
                for ni_idx,ni in enumerate(n):
                    if abs(ni) > 1e-12:
                        print("   Rotated State %4i: %12.8f"%(ni_idx,ni))
                rotations[fspace] = U
            print(" Final norm: %12.8f"%norm)
        
            ci.rotate_basis(rotations)
        delta_e = e0 - e_last
        e_last = e0
        if abs(delta_e) < 1e-8:
            print(" Converged BRDM iterations")
            break
    return ci_vector, e0
# }}}


if __name__=="__main__":
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)

    n_orb = 6
    U = 1.0
    beta = 1.0
    #n_cluster_states = 9
    cs_ratio = 1
    nel = n_orb//2

    h, g = get_hubbard_params(n_orb,beta,U,pbc=True)
    np.random.seed(2)
    tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
    h += tmp + tmp.T
    print(h)

    if 1:
        Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

    do_fci = 0
    if do_fci:
        # FCI
        from pyscf import gto, scf, ao2mo, fci, cc
        pyscf.lib.num_threads(1)
        mol = gto.M(verbose=3)
        mol.nelectron = n_orb
        # Setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
        # attribute) to be used in the post-HF calculations.  Without this parameter,
        # some post-HF method (particularly in the MO integral transformation) may
        # ignore the customized Hamiltonian if memory is not enough.
        mol.incore_anyway = True
        cisolver = fci.direct_spin1.FCI(mol)
        #e, ci = cisolver.kernel(h1, eri, h1.shape[1], 2, ecore=mol.energy_nuc())
        e, ci = cisolver.kernel(h, g, h.shape[1], mol.nelectron, ecore=0)
        print(" FCI:        %12.8f"%e)

    blocks = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    blocks = [[0],[1],[2],[3],[4],[5]]
    blocks = [[0,1,2],[3,4,5]]
    blocks = [range(nel),range(nel,2*nel)]
    n_blocks = len(blocks)
    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState(clusters)
    #ci_vector.init(((2,2),(0,0)))
    #ci_vector.init(((2,2),(2,2),(0,0)))
    #ci_vector.init(((2,2),(2,2),(0,0),(0,0)))
    #ci_vector.init(((5,5),(0,0)))
    #ci_vector.init(((1,1),(1,1),(1,1),(1,1),(1,1),(0,0),(0,0),(0,0),(0,0),(0,0)))
    #ci_vector.init(((1,1),(1,1),(1,1),(0,0),(0,0),(0,0)))
    #ci_vector.init(((2,1),(1,2)))
    ci_vector.init(((nel,nel),(0,0)))
    ci_vector.init(((nel-1,nel-1),(1,1)))
    ci_vector.init(((nel,nel-2),(0,2)))
    ci_vector.init(((nel-2,nel),(2,0)))


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
        ci.form_eigbasis_from_local_operator(opi,ratio=cs_ratio)


    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx)
        c.build_op_matrices()

    #ci_vector.expand_to_full_space()
    ci_vector.expand_each_fock_space()

    e_prev = 0
    thresh_conv = 1e-8
    ci_vector_ref = ci_vector.copy()
    e_last = 0
    #ci_vector.print_configs()
    #ci_vector, e0 = bc_cid(ci_vector_ref.copy(), clustered_ham, thresh_cipsi=1e-14)
    ci_vector, e0 = bc_cid_tucker(ci_vector_ref.copy(), clustered_ham, thresh_cipsi=1e-14, thresh_cipsi_conv=1e-8, thresh_tucker_conv = 1e-6, max_tucker_iter=20)
    civec = ci_vector.get_vector()
    ci_vector.clip(1e-14)
    ci_vector.print_configs()
    occ = nel
    vir = n_orb - occ
    civec = civec.reshape(civec.shape[0])
    ci_ab = civec[1:occ*occ*vir*vir+1].reshape(occ*occ,vir*vir)

    print(" CID: E0 = %12.8f  CI_DIM: %i" %(e0, len(ci_vector)))
    U,s,Vt = np.linalg.svd(ci_ab)
    print(s)
    print(ci_ab)
    print("Occupied Unitary")
    print(U)

