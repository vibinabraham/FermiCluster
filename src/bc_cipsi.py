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

def bc_cipsi_tucker(ci_vector, clustered_ham, selection="cipsi",
        thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_cipsi_conv=1e-8, max_cipsi_iter=30, 
        thresh_tucker_conv = 1e-6, max_tucker_iter=20, tucker_state_clip=None,hshift=1e-8,
        thresh_asci=0,nproc=None):
    """
    Run iterations of TP-CIPSI to make the tucker decomposition self-consistent
    """
# {{{

    
    t_conv = False
    if tucker_state_clip == None:
        tucker_state_clip = thresh_cipsi/10.0
    e_prev = 0
    e_last = 0
    ci_vector_ref = ci_vector.copy()
    for brdm_iter in range(max_tucker_iter):
   
        if selection == "cipsi":
            start = time.time()
            ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector_ref.copy(), clustered_ham, 
                    thresh_cipsi=thresh_cipsi, thresh_ci_clip=thresh_ci_clip, thresh_conv=thresh_cipsi_conv, max_iter=max_cipsi_iter,thresh_asci=thresh_asci,
                    nproc=nproc)
            end = time.time()
            e_curr = e2
            print(" CIPSI: E0 = %12.8f E2 = %12.8f CI_DIM: %-12i Time spent %-12.2f" %(e0, e2, len(ci_vector), end-start))
            
            pt_vector.add(ci_vector)
            print(" Reduce size of 1st order wavefunction")
            print(" Before:",len(pt_vector))
            pt_vector.clip(tucker_state_clip)
            pt_vector.normalize()
            print(" After:",len(pt_vector))
            ci_vector = pt_vector 
        elif selection == "heatbath":
            start = time.time()
            ci_vector, e0 = hb_tpsci(ci_vector_ref.copy(), clustered_ham, 
                    thresh_cipsi=thresh_cipsi, thresh_ci_clip=thresh_ci_clip, thresh_conv=thresh_cipsi_conv, max_iter=max_cipsi_iter,thresh_asci=thresh_asci,
                    nproc=nproc)
            end = time.time()
            pt_vector = ClusteredState(ci_vector.clusters)
            e_curr = e0
            e2 = 0
            print(" HB-TPSCI: E0 = %12.8f CI_DIM: %-12i Time spent %-12.2f" %(e0, len(ci_vector), end-start))

        
      
        if abs(e_prev-e_curr) < thresh_tucker_conv:
            print(" Converged BRDMs")
            t_conv = True
            break
        e_prev = e_curr
        for ci in clustered_ham.clusters:
            print()
            print(" --------------------------------------------------------")
            print(" Density matrix: Cluster ", ci)
            print()
            print(" Compute BRDM",flush=True)
            print(" Hshift = ",hshift)
            start = time.time()
            rdms = build_brdm(ci_vector, ci.idx)
            end = time.time()
            print(" done.",flush=True)
            print(" Time spent building BRDMs: %12.2f" %(end-start))
            #rdms = build_brdm(ci_vector, ci.idx)
            norm = 0
            entropy = 0
            rotations = {}
            for fspace,rdm in rdms.items():
               
                fspace_norm = 0
                fspace_entropy = 0
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
                fspace_norm = sum(n)
                for ni_idx,ni in enumerate(n):
                    if abs(ni/norm) > 1e-12:
                        fspace_entropy -= ni*np.log(ni/norm)/norm
                        entropy -=  ni*np.log(ni)
                        print("   Rotated State %4i:    %12.8f"%(ni_idx,ni), flush=True)
                print("   ----")
                print("   Entanglement entropy:  %12.8f" %fspace_entropy, flush=True) 
                print("   Norm:                  %12.8f" %fspace_norm, flush=True) 
                rotations[fspace] = U
            print(" Final entropy:.... %12.8f"%entropy)
            print(" Final norm:....... %12.8f"%norm)
            print(" --------------------------------------------------------", flush=True)
        
            start = time.time()
            ci.rotate_basis(rotations)
            end = time.time()
            print(" Time spent rotating cluster basis: %12.2f" %(end-start))
        delta_e = e0 - e_last
        e_last = e0
        if abs(delta_e) < 1e-8:
            print(" Converged BRDM iterations")
            break
    return ci_vector, pt_vector, e0, e2, t_conv
# }}}

#def bc_cipsi_1rdm(ci_vector, clustered_ham, h, g,
#        thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_cipsi_conv=1e-8, max_cipsi_iter=30, 
#        thresh_rdm_conv = 1e-6, max_rdm_iter=20,hshift=1e-8,thresh_asci=0):
#    """
#    Run iterations of TP-CIPSI to make the spin-averaged 1rdm potential self-consistent
#    """
## {{{
#
#    t_conv = False
#    e_prev = 0
#    e_last = 0
#    ci_vector_ref = ci_vector.copy()
#    for rdm_iter in range(max_rdm_iter):
#        ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector_ref.copy(), clustered_ham, 
#                thresh_cipsi=thresh_cipsi, thresh_ci_clip=thresh_ci_clip, thresh_conv=thresh_cipsi_conv, max_iter=max_cipsi_iter,thresh_asci=thresh_asci)
#        
#        print(" CIPSI: E0 = %12.8f E2 = %12.8f CI_DIM: %i" %(e0, e2, len(ci_vector)))
#      
#        if abs(e_prev-e2) < thresh_rdm_conv:
#            print(" Converged 1RDM")
#            t_conv = True
#            break
#        e_prev = e2
#        pt_vector.add(ci_vector)
#        print(" Reduce size of 1st order wavefunction")
#        print(" Before:",len(pt_vector))
#        pt_vector.normalize()
#        print(" After:",len(pt_vector))
#        print()
#        print(" Compute 1RDM",flush=True)
#        
#        # form 1rdm from current state
#        rdm_a, rdm_b = tools.build_1rdm(ci_vector)
#
#        print(" done.",flush=True)
#            
#        print(" Build cluster basis")
#        for ci_idx, ci in enumerate(clustered_ham.clusters):
#            assert(ci_idx == ci.idx)
#            #print(" Extract local operator for cluster",ci.idx)
#            #opi = clustered_ham.extract_local_operator(ci_idx)
#            print()
#            print()
#            print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
#            #ci.form_eigbasis_from_local_operator(opi,max_roots=1000)
#            
#            #ci.form_eigbasis_from_ints(h,g,max_roots=1000)
#            ci.form_eigbasis_from_ints(h,g,max_roots=1000, rdm1_a=rdm_a, rdm1_b=rdm_b)
#        
#        
#        #clustered_ham.add_ops_to_clusters()
#        print(" Build these local operators")
#        for c in clustered_ham.clusters:
#            print(" Build mats for cluster ",c.idx)
#            c.build_op_matrices()
#
#        delta_e = e0 - e_last
#        e_last = e0
#        if abs(delta_e) < 1e-8:
#            print(" Converged 1RDM iterations")
#            break
#    return ci_vector, pt_vector, e0, e2, t_conv
## }}}


def bc_cipsi(ci_vector, clustered_ham,  
    thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_conv=1e-8, max_iter=30, n_roots=1,thresh_asci=0,nproc=None):
# {{{
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
        start = time.time()
        if nproc==1:
            H = build_full_hamiltonian(clustered_ham, ci_vector)
        else:
            H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector, nproc=nproc)
        stop = time.time()
        print(" Time spent building Hamiltonian matrix: %12.2f" %(stop-start))
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
            kept_indices = ci_vector.clip(np.sqrt(thresh_ci_clip))
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

        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : ", len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : ", len(asci_vector))
        #asci_vector.normalize()

        print(" Compute Matrix Vector Product:", flush=True)
        start = time.time()
        if nproc==1:
            pt_vector = matvec1(clustered_ham, asci_vector)
        else:
            pt_vector = matvec1_parallel2(clustered_ham, asci_vector, nproc=nproc)
        stop = time.time()
        print(" Time spent in matvec: %12.2f" %( stop-start))
        
        pt_vector.prune_empty_fock_spaces()


        var = pt_vector.dot(pt_vector) - e0*e0
        print(" Variance:          %12.8f" % var,flush=True)
        tmp = ci_vector.dot(pt_vector)
        var = pt_vector.dot(pt_vector) - tmp*tmp
        print(" Variance Subspace: %12.8f" % var,flush=True)


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

        start = time.time()
        pt_vector.prune_empty_fock_spaces()
            
        #import cProfile
        #pr = cProfile.Profile()
        #pr.enable()
           
        if nproc==1:
            Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
        else:
            Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
        #pr.disable()
        #pr.print_stats(sort='time')
        end = time.time()
        print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)

        denom = 1/(e0 - Hd)
        pt_vector_v = pt_vector.get_vector()
        pt_vector_v.shape = (pt_vector_v.shape[0])

        e2 = np.multiply(denom,pt_vector_v)
        pt_vector.set_vector(e2)
        e2 = np.dot(pt_vector_v,e2)

        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2))

        print(" Choose which states to add to CI space", flush=True)

        start = time.time()
        for fockspace,configs in pt_vector.items():
            for config,coeff in configs.items():
                if coeff*coeff > thresh_cipsi:
                    if fockspace in ci_vector:
                        ci_vector[fockspace][config] = 0
                    else:
                        ci_vector.add_fockspace(fockspace)
                        ci_vector[fockspace][config] = 0
        end = time.time()
        print(" Time spent in finding new CI space: %12.2f" %(end - start), flush=True)

        delta_e = e0 - e_prev
        e_prev = e0
        if len(ci_vector) <= old_dim and abs(delta_e) < thresh_conv:
            print(" Converged")
            if thresh_asci > 0:
                print("\n Compute Final PT vector and correction with full variational space")
                start = time.time()
                if nproc==1:
                    pt_vector = matvec1(clustered_ham, ci_vector)
                else:
                    pt_vector = matvec1_parallel2(clustered_ham, ci_vector, nproc=nproc)
                stop = time.time()
                print(" Time spent in matvec: %12.2f" %(stop-start), flush=True)
                pt_vector.prune_empty_fock_spaces()

                var = pt_vector.dot(pt_vector) - e0*e0
                print(" Variance:          %12.8f" % var,flush=True)
                tmp = ci_vector.dot(pt_vector)
                var = pt_vector.dot(pt_vector) - tmp*tmp
                print(" Variance Subspace: %12.8f" % var,flush=True)


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
                    
                if nproc==1:
                    Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
                else:
                    Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
                #pr.disable()
                #pr.print_stats(sort='time')
                end = time.time()
                print(" Time spent in demonimator: %12.2f" %(end - start), flush=True)

                denom = 1/(e0 - Hd)
                pt_vector_v = pt_vector.get_vector()
                pt_vector_v.shape = (pt_vector_v.shape[0])

                e2 = np.multiply(denom,pt_vector_v)
                pt_vector.set_vector(e2)
                e2 = np.dot(pt_vector_v,e2)

                print(" PT2 Energy Correction = %12.8f" %e2, flush=True)
                print(" PT2 Energy Total      = %12.8f" %(e0+e2), flush=True)
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

# }}}


def hb_tpsci(ci_vector, clustered_ham, thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_conv=1e-8, max_iter=30, n_roots=1,thresh_asci=0,nproc=None):
# {{{
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
        start = time.time()
        if nproc==1:
            H = build_full_hamiltonian(clustered_ham, ci_vector)
        else:
            H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector, nproc=nproc)
        stop = time.time()
        print(" Time spent building Hamiltonian matrix: %12.2f" %(stop-start))
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
            kept_indices = ci_vector.clip(np.sqrt(thresh_ci_clip))
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
        print(" Perform Heat-Bath selection to find new configurations:",flush=True)
        start=time.time()
        pt_vector = heat_bath_search(clustered_ham, ci_vector, thresh_cipsi=thresh_cipsi, nproc=nproc)
        stop=time.time()
        print(" Number of new configurations found         : ", len(pt_vector))
        print(" Time spent in heat bath search: %12.2f" %(stop-start),flush=True)

        print(" Remove CI space from results")
        for fockspace,configs in pt_vector.items():
            if fockspace in ci_vector.fblocks():
                for config,coeff in list(configs.items()):
                    if config in ci_vector[fockspace]:
                        del pt_vector[fockspace][config]
        print(" Number of new configurations found (pruned): ", len(pt_vector))


        for fockspace,configs in ci_vector.items():
            if fockspace in pt_vector:
                for config,coeff in configs.items():
                    assert(config not in pt_vector[fockspace])

        print(" Norm of CI vector = %12.8f" %ci_vector.norm())

        print(" Add new states to CI space", flush=True)
        print(" Dimension of CI space     : ", len(ci_vector))
        start = time.time()
        pt_vector.zero()
        ci_vector.add(pt_vector)
        end = time.time()
        print(" Dimension of next CI space: ", len(ci_vector))
        print(" Time spent in finding new CI space: %12.2f" %(end - start), flush=True)

        start = time.time()

        delta_e = e0 - e_prev
        e_prev = e0
        if len(ci_vector) <= old_dim and abs(delta_e) < thresh_conv:
            print(" Converged")
            break
        print(" Next iteration CI space dimension", len(ci_vector))
    
    return ci_vector, e0

# }}}

def hosvd(ci_vector, clustered_ham, hshift=1e-8):
    """
    Peform HOSVD aka Tucker Decomposition of ClusteredState
    """
    for ci in clustered_ham.clusters:
        print()
        print(" --------------------------------------------------------")
        print(" Density matrix: Cluster ", ci)
        print()
        print(" Compute BRDM",flush=True)
        print(" Hshift = ",hshift)
        start = time.time()
        rdms = build_brdm(ci_vector, ci.idx)
        end = time.time()
        print(" done.",flush=True)
        print(" Time spent building BRDMs: %12.2f" %(end-start))
        #rdms = build_brdm(ci_vector, ci.idx)
        norm = 0
        entropy = 0
        rotations = {}
        for fspace,rdm in rdms.items():
           
            fspace_norm = 0
            fspace_entropy = 0
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
            fspace_norm = sum(n)
            for ni_idx,ni in enumerate(n):
                if abs(ni/norm) > 1e-12:
                    fspace_entropy -= ni*np.log(ni/norm)/norm
                    entropy -=  ni*np.log(ni)
                    print("   Rotated State %4i:    %12.8f"%(ni_idx,ni), flush=True)
            print("   ----")
            print("   Entanglement entropy:  %12.8f" %fspace_entropy, flush=True) 
            print("   Norm:                  %12.8f" %fspace_norm, flush=True) 
            rotations[fspace] = U
        print(" Final entropy:.... %12.8f"%entropy)
        print(" Final norm:....... %12.8f"%norm)
        print(" --------------------------------------------------------", flush=True)
    
        start = time.time()
        ci.rotate_basis(rotations)
        end = time.time()
        print(" Time spent rotating cluster basis: %12.2f" %(end-start))
    return




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
        dm_a = np.zeros((n_orb,n_orb))
        for i in range(n_orb//2):
            dm_a[i,i] = 1.0 
        print(" Density matrix: ")
        print(dm_a)
        dm_b = dm_a
    
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
        
        dm = dm_a + dm_b

        ci.form_eigbasis_from_ints(h,g,max_roots=1000, rdm1=dm)
    
    
    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx)
        c.build_op_matrices()
    
    # form 1rdm from reference state
    rdm_a, rdm_b = tools.build_1rdm(ci_vector)

    
    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()
    ci_vector.add_single_excitonic_states()
    ci_vector.print_configs()


    ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-5,
            thresh_ci_clip=1e-4, max_tucker_iter = 20)

    
