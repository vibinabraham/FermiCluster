import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import time
from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *

def bc_cipsi_tucker(ci_vector, clustered_ham, 
        thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_cipsi_conv=1e-8, max_cipsi_iter=30, 
        thresh_tucker_conv = 1e-6, max_tucker_iter=20, tucker_state_clip=None,hshift=1e-8):
    """
    Run iterations of TP-CIPSI to make the tucker decomposition self-consistent
    """
# {{{
    if tucker_state_clip == None:
        tucker_state_clip = thresh_cipsi/10.0
    e_prev = 0
    e_last = 0
    ci_vector_ref = ci_vector.copy()
    for brdm_iter in range(max_tucker_iter):
        ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector_ref.copy(), clustered_ham, 
                thresh_cipsi=thresh_cipsi, thresh_ci_clip=thresh_ci_clip, thresh_conv=thresh_cipsi_conv, max_iter=max_cipsi_iter)
        
        print(" CIPSI: E0 = %12.8f E2 = %12.8f CI_DIM: %i" %(e0, e2, len(ci_vector)))
      
        if abs(e_prev-e2) < thresh_tucker_conv:
            print(" Converged BRDMs")
            break
        e_prev = e2
        pt_vector.add(ci_vector)
        print(" Reduce size of 1st order wavefunction")
        print(" Before:",len(pt_vector))
        pt_vector.clip(tucker_state_clip)
        pt_vector.normalize()
        print(" After:",len(pt_vector))
        for ci in clustered_ham.clusters:
            print()
            print(" Compute BRDM",flush=True)
            print(" Hshift = ",hshift)
            rdms = build_brdm(pt_vector, ci.idx)
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

                if hshift != None:
                    """Adding cluster hamiltonian to RDM before diagonalization to make null space unique. """
                    Hci = ci.Hci[fspace]
                    n,U = np.linalg.eigh(rdm + hshift*Hci)
                    n = np.diag(U.T @ rdm @ U)
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
    return ci_vector, pt_vector, e0, e2
# }}}


def bc_cipsi(ci_vector, clustered_ham, thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_conv=1e-8, max_iter=30, n_roots=1):

    print(" Compute diagonal elements",flush=True)
    # compute local states energies
    precompute_cluster_basis_energies(clustered_ham)
    print(" done.",flush=True)

    pt_vector = ci_vector.copy()
    Hd_vector = ClusteredState(ci_vector.clusters)
    e_prev = 0
    for it in range(max_iter):
        print()
        print(" ===================================================================")
        print("     Selected CI Iteration: %4i epsilon: %12.8f" %(it,thresh_cipsi))
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

        old_dim = len(ci_vector)

        if thresh_ci_clip > 0:
            print(" Clip CI Vector: thresh = ", thresh_ci_clip)
            print(" Old CI Dim: ", len(ci_vector))
            kept_indices = ci_vector.clip(thresh_ci_clip)
            ci_vector.normalize()
            print(" New CI Dim: ", len(ci_vector))
            if len(ci_vector) < old_dim:
                H = H[:,kept_indices][kept_indices,:]
                print(" Diagonalize Clipped Hamiltonian Matrix:",flush=True)
                vguess = ci_vector.get_vector()
                e,v = scipy.sparse.linalg.eigsh(H,n_roots,v0=vguess,which='SA')
                #e,v = np.linalg.eigh(H)
                idx = e.argsort()
                e = e[idx]
                v = v[:,idx]
                v0 = v[:,0]
                e0 = e[0]
                print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))

                ci_vector.zero()
                ci_vector.set_vector(v0)


        #for i,j,k in ci_vector:
        #$    print(" iterator:", " Fock Space:", i, " Config:", j, " Coeff: %12.8f"%k)


        print(" Compute Matrix Vector Product:", flush=True)
        pt_vector = matvec1(clustered_ham, ci_vector)
        pt_vector.prune_empty_fock_spaces()
        #pt_vector.print()


        var = pt_vector.norm() - e0*e0
        print(" Variance: %12.8f" % var,flush=True)


        print(" Remove CI space from pt_vector vector")
        for fockspace,configs in pt_vector.items():
            if fockspace in ci_vector.fblocks():
                for config,coeff in list(configs.items()):
                    if config in ci_vector[fockspace]:
                        del pt_vector[fockspace][config]


        for fockspace,configs in ci_vector.items():
            if fockspace in pt_vector:
                for config,coeff in configs.items():
                    assert(config not in pt_vector[fockspace])

        print(" Norm of CI vector = %12.8f" %ci_vector.norm())
        print(" Dimension of CI space: ", len(ci_vector))
        print(" Dimension of PT space: ", len(pt_vector))
        print(" Compute Denominator",flush=True)
        #next_ci_vector = cp.deepcopy(ci_vector)
        # compute diagonal for PT2

        start = time.time()
        pt_vector.prune_empty_fock_spaces()
            
        #import cProfile
        #pr = cProfile.Profile()
        #pr.enable()
            
        Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
        #pr.disable()
        #pr.print_stats(sort='time')
        end = time.time()
        print(" Time spent in demonimator: ", end - start)

        denom = 1/(e0 - Hd)
        pt_vector_v = pt_vector.get_vector()
        pt_vector_v.shape = (pt_vector_v.shape[0])

        e2 = np.multiply(denom,pt_vector_v)
        pt_vector.set_vector(e2)
        e2 = np.dot(pt_vector_v,e2)

        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2))

        print(" Choose which states to add to CI space")

        for fockspace,configs in pt_vector.items():
            for config,coeff in configs.items():
                if coeff*coeff > thresh_cipsi:
                    if fockspace in ci_vector:
                        ci_vector[fockspace][config] = 0
                    else:
                        ci_vector.add_fockspace(fockspace)
                        ci_vector[fockspace][config] = 0
        delta_e = e0 - e_prev
        e_prev = e0
        if len(ci_vector) <= old_dim and abs(delta_e) < thresh_conv:
            print(" Converged")
            break
        print(" Next iteration CI space dimension", len(ci_vector))
    #    print(" Do CMF:")
    #    for ci_idx, ci in enumerate(clusters):
    #        assert(ci_idx == ci.idx)
    #        print(" Extract local operator for cluster",ci.idx)
    #        opi = build_effective_operator(ci_idx, clustered_ham, ci_vector)
    #        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
    #        ci.form_eigbasis_from_local_operator(opi,max_roots=1000)
    #        exit()
    return ci_vector, pt_vector, e0, e0+e2



if __name__ == "__main__":
    import pyscf
    ttt = time.time()
    
    n_orb = 6
    U = 2.
    beta = 1.0

    h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
    np.random.seed(2)
    tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
    h += tmp + tmp.T
    
    if 1:
        Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)
    
    do_fci = 1
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
    
    blocks = [[0],[1,2,3,4],[5]]
    n_blocks = len(blocks)
    clusters = []
    
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector = ClusteredState(clusters)
    ci_vector.init(((1,1),(2,2),(0,0)))
   

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
        #print(" Extract local operator for cluster",ci.idx)
        #opi = clustered_ham.extract_local_operator(ci_idx)
        print()
        print()
        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
        #ci.form_eigbasis_from_local_operator(opi,max_roots=1000)
        ci.form_eigbasis_from_ints(h,g,max_roots=1000)
    
    
    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx)
        c.build_op_matrices()
    
    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()


    ci_vector, pt_vector, e0, e2 = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-5,
            thresh_ci_clip=1e-4, max_tucker_iter = 20)

    
