import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import time
import pickle
from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *




def system_setup(h, g, ecore, blocks, max_roots=1000, max_nelec=None, min_nelec=None):
# {{{
    n_blocks = len(blocks)

    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    print(" Build cluster basis")
    for ci_idx, ci in enumerate(clusters):
        assert(ci_idx == ci.idx)
        #print(" Extract local operator for cluster",ci.idx)
        #opi = clustered_ham.extract_local_operator(ci_idx)
        #print()
        print()
        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
        #ci.form_eigbasis_from_local_operator(opi,max_roots=1000,s2_shift=s2_shift)
        ci.form_eigbasis_from_ints(h,g,
                max_roots=max_roots, 
                max_nelec=max_nelec,
                min_nelec=min_nelec,
                ecore=ecore)


    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx,flush=True)
        c.build_op_matrices()
        c.build_local_terms(h,g)
    
    return clusters, clustered_ham
# }}}


def bc_cipsi_tucker(ci_vector, clustered_ham, 
        selection           = "cipsi",
        thresh_cipsi        = 1e-4, 
        thresh_ci_clip      = 1e-5, 
        thresh_cipsi_conv   = 1e-8, 
        max_cipsi_iter      = 30, 
        thresh_tucker_conv  = 1e-6, 
        max_tucker_iter     = 20, 
        tucker_state_clip   = None,
        hshift              = 1e-8,
        thresh_asci         = 0, 
        thresh_search       = None, 
        pt_type             = 'en',
        nbody_limit         = 4, 
        tucker_conv_target  = 2, 
        nproc               = None):
    """
    Run iterations of TP-CIPSI to make the tucker decomposition self-consistent
   
    thresh_tucker_conv  :
    thresh_cipsi        :   include qspace configurations into pspace that have probabilities larger than this value
    thresh_ci_clip      :   drop pspace configs whose variational solution yields probabilities smaller than this value
    thresh_cipsi_conv   :   stop selected CI when delta E is smaller than this value
    thresh_asci         :   only consider couplings to pspace configs with probabilities larger than this value
    thresh_search       :   delete couplings to pspace configs
                                default: thresh_cipsi^1/2 / 1000
    pt_type             :   Which denominator to use? Epstein-Nesbitt (en) or Moller-Plesset-like (mp)
    tucker_conv_target  :   Which energy should we use to determine convergence? 
                                0 = variational energy
                                2 = pt2 energy
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
                                                        pt_type         = pt_type,
                                                        thresh_cipsi    = thresh_cipsi, 
                                                        thresh_ci_clip  = thresh_ci_clip, 
                                                        thresh_conv     = thresh_cipsi_conv, 
                                                        max_iter        = max_cipsi_iter,
                                                        thresh_asci     = thresh_asci,
                                                        nbody_limit     = nbody_limit, 
                                                        thresh_search   = thresh_search, 
                                                        nproc           = nproc)
            end = time.time()
            if tucker_conv_target == 0:
                e_curr = e0
            elif tucker_conv_target == 2:
                e_curr = e2
            else:
                print(" wrong value for tucker_conv_target")
                exit()
            print(" TPSCI: E0 = %12.8f E2 = %12.8f CI_DIM: %-12i Time spent %-12.2f" %(e0, e2, len(ci_vector), end-start))
            
        elif selection == "heatbath":
            start = time.time()
            ci_vector, e0 = hb_tpsci(ci_vector_ref.copy(), clustered_ham, 
                    thresh_cipsi=thresh_cipsi, thresh_ci_clip=thresh_ci_clip, nbody_limit=nbody_limit, 
                    thresh_conv=thresh_cipsi_conv, max_iter=max_cipsi_iter,thresh_asci=thresh_asci,
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
        delta_e = e0 - e_last
        e_last = e0
        if abs(delta_e) < 1e-8:
            print(" Converged BRDM iterations")
            break
        
        # do the Tucker decomposition
        if selection == "cipsi":
            pt_vector.add(ci_vector)
            print(" Reduce size of 1st order wavefunction")
            print(" Before:",len(pt_vector))
            pt_vector.clip(tucker_state_clip)
            pt_vector.normalize()
            print(" After:",len(pt_vector))

            hosvd(pt_vector, clustered_ham, hshift=hshift)
        
        elif selection == "heatbath":
            hosvd(ci_vector, clustered_ham, hshift=hshift)


        print(" Ensure TDMs are still contiguous:")
        for ci in clustered_ham.clusters:
            print(ci)
            for o in ci.ops:
                for fock in ci.ops[o]:
                    if ci.ops[o][fock].data.contiguous == False:
                        print(" Rearrange data for %5s :" %o, fock)
                        ci.ops[o][fock] = np.ascontiguousarray(ci.ops[o][fock])
    
        
        print(" Saving Hamiltonian to disk",flush=True)
        hamiltonian_file = open('hamiltonian_file_tucker', 'wb')
        pickle.dump(clustered_ham, hamiltonian_file)
        print(" Done.",flush=True)


    return ci_vector, pt_vector, e0, e2, t_conv
# }}}


def bc_cipsi(ci_vector, clustered_ham,  
    thresh_cipsi    = 1e-4, 
    thresh_ci_clip  = 1e-5, 
    thresh_conv     = 1e-8, 
    max_iter        = 30, 
    n_roots         = 1,
    thresh_asci     = 0,
    nbody_limit     = 4, 
    pt_type         = 'en',
    thresh_search   = None, 
    nproc=None):
    """
    thresh_cipsi    :   include qspace configurations into pspace that have probabilities larger than this value
    thresh_ci_clip  :   drop pspace configs whose variational solution yields probabilities smaller than this value
    thresh_conv     :   stop selected CI when delta E is smaller than this value
    thresh_asci     :   only consider couplings to pspace configs with probabilities larger than this value
    thresh_search   :   delete couplings to pspace configs
                        default: thresh_cipsi^1/2 / 1000
    nbody_limit     :   only compute up to n-body interactions when searching for new configs
    """
# {{{
  
    if thresh_search == None:
        thresh_search = np.sqrt(thresh_cipsi / 1000) 
        print(" No thresh_search defined, choosing default: ", thresh_search)
   
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
        
        delta_e = e0 - e_prev
        e_prev = e0
        if len(ci_vector) <= old_dim and abs(delta_e) < thresh_conv:
            print(" Converged")
            break
        print(" Next iteration CI space dimension", len(ci_vector))
    

        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : ", len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()

        print(" Compute Matrix Vector Product:", flush=True)

        profile = 0
        if profile:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()

        start = time.time()
        if nbody_limit != 4:
            print(" Warning: nbody_limit set to %4i, resulting PT energies are meaningless" %nbody_limit)

        if nproc==1:
            pt_vector = matvec1(clustered_ham, asci_vector, thresh_search=thresh_search, nbody_limit=nbody_limit)
        else:
            pt_vector = matvec1_parallel3(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        stop = time.time()
        print(" Time spent in matvec: %12.2f" %( stop-start))
        #exit()
        
        if profile:
            pr.disable()
            pr.print_stats(sort='time')
        
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
        #exit()
        pt_vector.prune_empty_fock_spaces()
            
        #import cProfile
        #pr = cProfile.Profile()
        #pr.enable()


        # Build Denominator
        if pt_type == 'en':
            start = time.time()
            if nproc==1:
                Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
            else:
                Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
            #pr.disable()
            #pr.print_stats(sort='time')
            end = time.time()
            print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
            
            denom = 1/(e0 - Hd)
        elif pt_type == 'mp':
            start = time.time()
            # get barycentric MP zeroth order energy
            e0_mp = 0
            for f,c,v in ci_vector:
                for ci in clustered_ham.clusters:
                    e0_mp += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]] * v * v
            
            print(" Zeroth-order MP energy: %12.8f" %e0_mp, flush=True)

            #   This is not really MP once we have rotated away from the CMF basis.
            #   H = F + (H - F), where F = sum_I F(I)
            #
            #   After Tucker basis, we just use the diagonal of this fock operator. 
            #   Not ideal perhaps, but better than nothing at this stage
            denom = np.zeros(len(pt_vector))
            idx = 0
            for f,c,v in pt_vector:
                e0_X = 0
                for ci in clustered_ham.clusters:
                    e0_X += ci.ops['H_mf'][(f[ci.idx],f[ci.idx])][c[ci.idx],c[ci.idx]]
                denom[idx] = 1/(e0_mp - e0_X)
                idx += 1
            end = time.time()
            print(" Time spent in demonimator: %12.2f" %( end - start), flush=True)
        
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
        #exit()


    return ci_vector, pt_vector, e0, e0+e2

# }}}


def hb_tpsci(ci_vector, clustered_ham, 
        thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_conv=1e-8, max_iter=30, n_roots=1, thresh_asci=0,
        nbody_limit=4, 
        nproc=None):
# {{{
  
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
        
        ci_vector.print()
        
        print(" Perform Heat-Bath selection to find new configurations:",flush=True)
        start=time.time()
        if nproc==1:
            pt_vector = matvec1(clustered_ham, ci_vector, thresh_search=thresh_search, nbody_limit=nbody_limit)
        else:
            pt_vector = matvec1_parallel3(clustered_ham, ci_vector, nproc=nproc, thresh_search=thresh_cipsi, 
                    nbody_limit=nbody_limit, thresh_asci=thresh_asci)
        #pt_vector = heat_bath_search(clustered_ham, ci_vector, thresh_cipsi=thresh_cipsi, nproc=nproc)
        stop=time.time()
        print(" Number of new configurations found         : ", len(pt_vector))
        print(" Time spent in heat bath search: %12.2f" %(stop-start),flush=True)
        
        print(" Remove CI space from results")
        for fockspace,configs in pt_vector.items():
            if fockspace in ci_vector.fblocks():
                for config,coeff in list(configs.items()):
                    if config in ci_vector[fockspace]:
                        del pt_vector[fockspace][config]
        pt_vector.prune_empty_fock_spaces()
        print(" Number of new configurations found (pruned): ", len(pt_vector))


        for fockspace,configs in ci_vector.items():
            if fockspace in pt_vector:
                for config,coeff in configs.items():
                    assert(config not in pt_vector[fockspace])

        pt_vector.print()

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
# {{{
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
                Hlocal = ci.ops['H_mf'][(fspace,fspace)]
                n,U = np.linalg.eigh(rdm + hshift*Hlocal)
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
# }}}



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

    
