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

from cmf import *
from matvec import *
from ham_build import *
from nbtucker_sparse  import *


def system_setup(h, g, ecore, blocks, init_fspace,
        max_roots       = 1000, 
        delta_elec      = None,
        cmf_maxiter     = 10,
        cmf_thresh      = 1e-8,
        cmf_dm_guess    = None,     #initial guess density matrices  for cmf tuple(alpha, beta)
        cmf_diis        = False,
        cmf_diis_start  = 1,
        cmf_max_diis    = 6
        ):
# {{{
    
    """
    If an input list of Cluster objects is provided for clusters_in, then the CMF will be restricted to that space spanned by that
    current basis
    """
    print(" System setup option:")
    print("     |init_fspace    : ", init_fspace)
    print("     |max_roots      : ", max_roots)
    print("     |delta_elec     : ", delta_elec)
    print("     |cmf_diis       : ", cmf_diis)
    print("     |cmf_maxiter    : ", cmf_maxiter)
    print("     |cmf_dm_guess   : ", cmf_dm_guess != None)
    print("     |cmf_thresh     : ", cmf_thresh)
    print("     |cmf_diis_start : ", cmf_diis_start)
    print("     |cmf_max_diis   : ", cmf_max_diis)
    print("     |Ecore          : ", ecore)
    n_blocks = len(blocks)
    clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    clustered_ham.h = h
    clustered_ham.g = g

    ci_vector = ClusteredState()
    ci_vector.init(clusters, init_fspace)

    if cmf_maxiter > 0:
        e_cmf, cmf_conv, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, 
                diis        = cmf_diis,
                dm_guess    = cmf_dm_guess,
                diis_start  = cmf_diis_start,
                max_diis    = cmf_max_diis,
                thresh      = cmf_thresh,
                max_iter    = cmf_maxiter)
    else:
        rdm_a = np.zeros(h.shape)
        rdm_b = np.zeros(h.shape)
        e_cmf = 0
        cmf_conv = False
    

    # build cluster basis and operator matrices using CMF optimized density matrices
            
    for ci_idx, ci in enumerate(clusters):
        if delta_elec != None:
            fspaces_i = init_fspace[ci_idx]
            fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_elec) )
        else:
            fspaces_i = ci.possible_fockspaces()
    
        print()
        print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
        ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=max_roots, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=ecore)
        
        print(" Build operator matrices for cluster ",ci.idx)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)

    
    return clusters, clustered_ham, ci_vector, (e_cmf, rdm_a, rdm_b, cmf_conv)
# }}}


def tpsci_tucker(ci_vector, clustered_ham, 
        selection           = "cipsi",
        thresh_cipsi        = 1e-4, 
        thresh_ci_clip      = 1e-5, 
        thresh_asci         = 0, 
        thresh_cipsi_conv   = 1e-8, 
        thresh_tucker_conv  = 1e-6, 
        thresh_search       = 0, 
        max_cipsi_iter      = 30, 
        max_tucker_iter     = 20, 
        tucker_state_clip   = None,
        tucker_truncate     = -1,
        hshift              = 1e-8,
        pt_type             = 'en',
        nbody_limit         = 4, 
        shared_mem          = 1e9,
        batch_size          = 1,
        tucker_conv_target  = 0,
        matvec              = 4,
        chk_file            = None,
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
    shared_mem          :   How much memory to allocate for shared object store for holding clustered_ham - only works with matvec4
    matvec              :   Which version of matvec to use? [1:4]
    tucker_state_clip   :   Delete PT1 coefficients smaller than this before adding to vector to perform HOSVD
    """
# {{{

    print(" Tucker optimization options:")
    print("     |selection          : ", selection          )
    print("     |thresh_cipsi       : ", thresh_cipsi       )
    print("     |thresh_ci_clip     : ", thresh_ci_clip     )
    print("     |thresh_cipsi_conv  : ", thresh_cipsi_conv  )
    print("     |max_cipsi_iter     : ", max_cipsi_iter     )
    print("     |thresh_tucker_conv : ", thresh_tucker_conv )
    print("     |max_tucker_iter    : ", max_tucker_iter    )
    print("     |tucker_state_clip  : ", tucker_state_clip  )
    print("     |tucker_truncate    : ", tucker_truncate    )
    print("     |hshift             : ", hshift             )
    print("     |thresh_asci        : ", thresh_asci        )
    print("     |thresh_search      : ", thresh_search      )
    print("     |pt_type            : ", pt_type            )
    print("     |nbody_limit        : ", nbody_limit        )
    print("     |tucker_conv_target : ", tucker_conv_target )
    print("     |nproc              : ", nproc              )
    
    t_conv = False
    if tucker_state_clip == None:
        tucker_state_clip = thresh_cipsi/10.0
    e_prev = 0
    e_last = 0
    ci_vector_ref = ci_vector.copy()
    for brdm_iter in range(max_tucker_iter+1):
   
        if selection == "cipsi":
            start = time.time()
            ci_vector, pt_vector, e0, e2 = tp_cipsi(ci_vector_ref.copy(), clustered_ham,
                                                        pt_type         = pt_type,
                                                        thresh_cipsi    = thresh_cipsi, 
                                                        thresh_ci_clip  = thresh_ci_clip, 
                                                        thresh_conv     = thresh_cipsi_conv, 
                                                        max_iter        = max_cipsi_iter,
                                                        thresh_asci     = thresh_asci,
                                                        nbody_limit     = nbody_limit,
                                                        matvec          = matvec,
                                                        batch_size      = batch_size,
                                                        shared_mem      = shared_mem,
                                                        thresh_search   = thresh_search, 
                                                        nproc           = nproc)
            end = time.time()
            
            ecore = clustered_ham.core_energy
            if tucker_conv_target == 0:
                e_curr = e0
            elif tucker_conv_target == 2:
                e_curr = e2
            else:
                print(" wrong value for tucker_conv_target")
                exit()
            print(" TPSCI: E0 = %12.8f E2 = %16.8f CI_DIM: %-12i Time spent %-12.1f" %(e0+ecore, e2+ecore, len(ci_vector), end-start))
            
        elif selection == "heatbath":
            start = time.time()
            ci_vector, e0 = tp_hbci(ci_vector_ref.copy(), clustered_ham, 
                                                        thresh_cipsi    = thresh_cipsi, 
                                                        thresh_ci_clip  = thresh_ci_clip, 
                                                        thresh_conv     = thresh_cipsi_conv, 
                                                        max_iter        = max_cipsi_iter,
                                                        thresh_asci     = thresh_asci,
                                                        nbody_limit     = nbody_limit,
                                                        matvec          = matvec,
                                                        batch_size      = batch_size,
                                                        shared_mem      = shared_mem,
                                                        thresh_search   = thresh_search, 
                                                        nproc           = nproc)
            end = time.time()
            pt_vector = ClusteredState()
            e_curr = e0
            e2 = 0
            ecore = clustered_ham.core_energy
            print(" HB-TPSCI: E0 = %16.8f CI_DIM: %-12i Time spent %-12.2f" %(e0+ecore, len(ci_vector), end-start))

        
      
        if abs(e_prev-e_curr) < thresh_tucker_conv:
            print(" Converged: Tucker")
            t_conv = True
            break
        elif brdm_iter >= max_tucker_iter:
            print(" Maxcycles: Tucker")
            t_conv = False 
            break

        e_prev = e_curr
        
        # do the Tucker decomposition
        if selection == "cipsi":
            print(" Reduce size of 1st order wavefunction",flush=True)
            print(" Before:",len(pt_vector))
            pt_vector.clip(tucker_state_clip)
            pt_vector.add(ci_vector)
            pt_vector.normalize()
            print(" After:",len(pt_vector),flush=True)

            hosvd(pt_vector, clustered_ham, hshift=hshift, truncate=tucker_truncate)
        
        elif selection == "heatbath":
            hosvd(ci_vector, clustered_ham, hshift=hshift, truncate=tucker_truncate)

    
        # Should we rebuild the operator matrices after rotating basis?
        if 0:
            h = clustered_ham.h
            g = clustered_ham.g
            for ci in clustered_ham.clusters:
                print(" Build operator matrices for cluster ",ci.idx)
                ci.build_op_matrices()
                ci.build_local_terms(h,g)

        print(" Ensure TDMs are still contiguous:", flush=True)
        start = time.time()
        for ci in clustered_ham.clusters:
            print(" ", ci)
            for o in ci.ops:
                for fock in ci.ops[o]:
                    if ci.ops[o][fock].data.contiguous == False:
                        #print(" Rearrange data for %5s :" %o, fock)
                        ci.ops[o][fock] = np.ascontiguousarray(ci.ops[o][fock])
        stop = time.time()
        print(" Time spent making operators contiguous: %12.2f" %( stop-start))
       
        if chk_file != None:
            print(" Saving Hamiltonian to disk",flush=True)
            file = open("%s_ham"%chk_file, 'wb')
            pickle.dump(clustered_ham, file)
            print(" Done.",flush=True)
            #print(" Saving wavefunction to disk",flush=True)
            #file = open("%s_vec"%chk_file, 'wb')
            #pickle.dump(ci_vector, file)
            print(" Done.",flush=True)


    return ci_vector, pt_vector, e0, e2, t_conv
# }}}


def tp_cipsi(ci_vector, clustered_ham,  
    thresh_cipsi    = 1e-4, 
    thresh_ci_clip  = 1e-5, 
    thresh_conv     = 1e-8, 
    max_iter        = 30, 
    n_roots         = 1,
    thresh_asci     = 0,
    nbody_limit     = 4, 
    pt_type         = 'en',
    thresh_search   = 0, 
    shared_mem      = 1e9,
    batch_size      = 1,
    matvec          = 4,
    nproc           = None
    ):
    """
    thresh_cipsi    :   include qspace configurations into pspace that have probabilities larger than this value
    thresh_ci_clip  :   drop pspace configs whose variational solution yields probabilities smaller than this value
    thresh_conv     :   stop selected CI when delta E is smaller than this value
    thresh_asci     :   only consider couplings to pspace configs with probabilities larger than this value
    thresh_search   :   delete couplings to pspace configs
                        default: thresh_cipsi^1/2 / 1000
    nbody_limit     :   only compute up to n-body interactions when searching for new configs
    shared_mem      :   How much memory to allocate for shared object store for holding clustered_ham - only works with matvec4
    matvec          :   Which version of matvec to use? [1:4]
    """
# {{{

    print()
    print(" TPSCI options: ")
    print("     |thresh_cipsi   : ", thresh_cipsi   )
    print("     |thresh_ci_clip : ", thresh_ci_clip )
    print("     |thresh_conv    : ", thresh_conv    )
    print("     |thresh_search  : ", thresh_search  )
    print("     |max_iter       : ", max_iter       )
    print("     |n_roots        : ", n_roots        )
    print("     |thresh_asci    : ", thresh_asci    )
    print("     |nbody_limit    : ", nbody_limit    )
    print("     |pt_type        : ", pt_type        )
    print("     |nproc          : ", nproc          )

    
    pt_vector = ci_vector.copy()
    #Hd_vector = ClusteredState(ci_vector.clusters)
    Hd_vector = ClusteredState()
    e_prev = 0
    for it in range(max_iter+1):
        print()
        print(" ===================================================================")
        print("     Selected CI Iteration: %4i epsilon: %12.8f" %(it,thresh_cipsi))
        print(" ===================================================================")
        print(" Build full Hamiltonian",flush=True)
        start = time.time()
        
        if it>0:
            H = grow_hamiltonian_parallel(H, clustered_ham, ci_vector, ci_vector_old)
        else:
            H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector, nproc=nproc)
        
        ci_vector_old = ci_vector.copy()
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
                ci_vector_old = ci_vector.copy()
        
        ecore = clustered_ham.core_energy
        print(" Core energy: %16.12f" %ecore)
        #print(" TPSCI Iter %3i Elec Energy:  %12.8f Total Energy:  %12.8f  CI Dim: %4i "%(it, e[0].real,e[0].real+ecore,len(ci_vector)))
        #print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))
        print(" TPSCI Iter %3i:                     %12.8f  CI Dim: %4i "%(it, e[0].real,len(ci_vector)))
        ci_vector.print()
        
        delta_e = e0 - e_prev
        e_prev = e0
        if len(ci_vector) <= old_dim and abs(delta_e) < thresh_conv:
            print(" Converged: TPSCI")
            break
        print(" Next iteration CI space dimension", len(ci_vector))
    

        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
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

        if matvec==1:
            pt_vector = matvec1_parallel1(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==2:
            pt_vector = matvec1_parallel2(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==3:
            pt_vector = matvec1_parallel3(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==4:
            pt_vector = matvec1_parallel4(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit,
                    shared_mem=shared_mem, batch_size=batch_size)
        stop = time.time()
        print(" Time spent in matvec: %12.2f" %( stop-start))
        #exit()
        
      
        # Compute the energy of the zeroth-order energy using matvec. 
        # This gives us some indication how accurate the approximations 
        # (asci_vector, and thresh_search) are
        e0_curr = ci_vector.dot(pt_vector)/asci_vector.dot(asci_vector) 
        print(" Zeroth-order energy: %12.8f Error in E0: %12.8f" %(e0_curr, e0_curr - e0)) 

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
        if len(pt_vector) == 0:
            print("No more connecting config found")
            break
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
            print(" Time spent in denomimator: %12.2f" %( end - start), flush=True)
            
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
            print(" Time spent in denomimator: %12.2f" %( end - start), flush=True)
        
        pt_vector_v = pt_vector.get_vector()
        pt_vector_v.shape = (pt_vector_v.shape[0])

        e2 = np.multiply(denom,pt_vector_v)
        pt_vector.set_vector(e2)
        e2 = np.dot(pt_vector_v,e2)

        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2))
        
        if it >= max_iter:
            print(" Maxcycles: TPSCI")
            break

        print(" Choose which states to add to CI space", flush=True)
        for fockspace,configs in pt_vector.items():
            for config,coeff in configs.items():
                if coeff*coeff > thresh_cipsi:
                    if fockspace in ci_vector:
                        ci_vector[fockspace][config] = 0
                    else:
                        ci_vector.add_fockspace(fockspace)
                        ci_vector[fockspace][config] = 0
        
        
        print(" Dimension of next CI space: ", len(ci_vector))

    return ci_vector, pt_vector, e0, e0+e2

# }}}


def tp_hbci(ci_vector, clustered_ham, 
        thresh_cipsi    = 1e-4, 
        thresh_ci_clip  = 1e-5, 
        thresh_conv     = 1e-8, 
        max_iter        = 30, 
        n_roots         = 1, 
        thresh_asci     = 0,
        nbody_limit     = 4, 
        thresh_search   = 0, 
        shared_mem      = 1e9,
        batch_size      = 1,
        matvec          = 4,
        nproc=None):
# {{{
    
    print()
    print(" HB-TPSCI options: ")
    print("     |thresh_cipsi   : ", thresh_cipsi   )
    print("     |thresh_ci_clip : ", thresh_ci_clip )
    print("     |thresh_conv    : ", thresh_conv    )
    print("     |max_iter       : ", max_iter       )
    print("     |n_roots        : ", n_roots        )
    print("     |thresh_asci    : ", thresh_asci    )
    print("     |nbody_limit    : ", nbody_limit    )
    print("     |nproc          : ", nproc          )
  
    pt_vector = ci_vector.copy()
    Hd_vector = ClusteredState()
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
            H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector, nproc=nproc)
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
        
        asci_vector = ci_vector.copy()
        print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
        print(" CI Dim          : %8i" % len(asci_vector))
        kept_indices = asci_vector.clip(thresh_asci)
        print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))
        #asci_vector.normalize()
        
        print(" Perform Heat-Bath selection to find new configurations:",flush=True)
        start=time.time()
        if matvec==1:
            pt_vector = matvec1_parallel1(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==2:
            pt_vector = matvec1_parallel2(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==3:
            pt_vector = matvec1_parallel3(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
        elif matvec==4:
            pt_vector = matvec1_parallel4(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit,
                    shared_mem=shared_mem, batch_size=batch_size)
        
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


def hosvd(ci_vector, clustered_ham, hshift=1e-8, truncate=-1):
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
        rdms = build_brdm(ci_vector, ci)
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
                #"""Adding cluster hamiltonian to RDM before diagonalization to make null space unique. """
                #Hlocal = ci.ops['H_mf'][(fspace,fspace)]
                #n,U = np.linalg.eigh(rdm + hshift*Hlocal)
                #n = np.diag(U.T @ rdm @ U)
                #idx = n.argsort()[::-1]
                #n = n[idx]
                #U = U[:,idx]

                
                """Adding cluster hamiltonian to RDM before diagonalization to make null space unique. """
                n,U = np.linalg.eigh(rdm)
                idx = n.argsort()[::-1]
                n = n[idx]
                U = U[:,idx]


                # Either truncate the unoccupied cluster states, or remix them with a hamiltonian to be unique
                if truncate < 0:
                    remix = []
                    for ni in range(n.shape[0]):
                        if n[ni] < 1e-8:
                            remix.append(ni)
                    U2 = U[:,remix]
                    Hlocal = U2.T @ ci.ops['H_mf'][(fspace,fspace)] @ U2
                    n2,v2 = np.linalg.eigh(Hlocal)
                    U2 = U2@v2
                    idx = n2.argsort()
                    n2 = n2[idx]
                    U2 = U2[:,idx]
                    
                    U[:,remix] = U2
                else:
                    keep = []
                    for ni in range(n.shape[0]):
                        if abs(n[ni]) > truncate:
                            keep.append(ni)
                    print(" Truncated Tucker space. Starting: %5i Ending: %5i" %(n.shape[0],len(keep)))
                    U = U[:,keep]
            
                if U.shape[1] > 0:
                    assert(np.amax(np.abs(U.T@U - np.eye(U.shape[1]))) < 1e-14)
           
            
            n = np.diag(U.T @ rdm @ U)
            Elocal = np.diag(U.T @ ci.ops['H_mf'][(fspace,fspace)] @ U)
            ## check orthogonality
            #if np.amax(np.abs(U.T@U - np.eye(U.shape[1]))) > 1e-12:
            #    S = U.T @ U
            #    S = scipy.linalg.inv( scipy.linalg.sqrtm(S))
            #    U = U@S
            #    print(" Warning: had to correct orthogonality", np.amax(np.abs(U.T@U - np.eye(U.shape[1]))))
            #    assert(np.amax(np.abs(U.T@U - np.eye(U.shape[1]))) < 1e-14)
            
            norm += sum(n)
            fspace_norm = sum(n)
            print("                 %4s:    %12s    %12s"%('','Population','Energy'), flush=True)
            for ni_idx,ni in enumerate(n):
                if abs(ni/norm) > 1e-18:
                    fspace_entropy -= ni*np.log(ni/norm)/norm
                    entropy -=  ni*np.log(ni)
                    print("   Rotated State %4i:    %12.8f    %12.8f"%(ni_idx,ni,Elocal[ni_idx]), flush=True)
            print("   ----")
            print("   Entanglement entropy:  %12.8f" %fspace_entropy, flush=True) 
            print("   Norm:                  %12.8f" %fspace_norm, flush=True) 
            #print(" det(U): %16.12f"%(np.linalg.det(U)))
            rotations[fspace] = U
        print(" Final entropy:.... %12.8f"%entropy)
        print(" Final norm:....... %12.8f"%norm)
        print(" --------------------------------------------------------", flush=True)
    
        start = time.time()
        ci.rotate_basis(rotations)
        ci.check_basis_orthogonality()
        end = time.time()
        print(" Time spent rotating cluster basis: %12.2f" %(end-start))
    return
# }}}


def ex_tp_cipsi(ci_vector, clustered_ham,  
    thresh_cipsi    = 1e-4, 
    thresh_conv     = 1e-8, 
    max_iter        = 30, 
    n_roots         = 3,
    thresh_asci     = 0,
    nbody_limit     = 4, 
    pt_type         = 'en',
    thresh_search   = 0, 
    shared_mem      = 1e9,
    batch_size      = 1,
    matvec          = 4,
    nproc           = None
    ):
    """
    +====================================================================+
                               Excited State TPSCI
    +====================================================================+
    
    ci_vector: the configutation space for all the needed roots. this is tricky since a pre computation of a CIS type calculation
        needs to be done for this. see test/excited_test.py


    thresh_cipsi    :   include qspace configurations into pspace that have probabilities larger than this value
    thresh_conv     :   stop selected CI when delta E is smaller than this value
    thresh_asci     :   only consider couplings to pspace configs with probabilities larger than this value
    thresh_search   :   delete couplings to pspace configs
                        default: thresh_cipsi^1/2 / 1000
    nbody_limit     :   only compute up to n-body interactions when searching for new configs
    shared_mem      :   How much memory to allocate for shared object store for holding clustered_ham - only works with matvec4
    matvec          :   Which version of matvec to use? [1:4]
    """
# {{{

    print()
    print(" Excited TPSCI options: ")
    print("     |thresh_cipsi   : ", thresh_cipsi   )
    print("     |thresh_conv    : ", thresh_conv    )
    print("     |thresh_search  : ", thresh_search  )
    print("     |max_iter       : ", max_iter       )
    print("     |n_roots        : ", n_roots        )
    print("     |thresh_asci    : ", thresh_asci    )
    print("     |nbody_limit    : ", nbody_limit    )
    print("     |pt_type        : ", pt_type        )
    print("     |nproc          : ", nproc          )

    ecore = clustered_ham.core_energy
    print(" Core energy: %16.12f" %ecore)

    
    pt_vector = ci_vector.copy()
    #Hd_vector = ClusteredState(ci_vector.clusters)
    Hd_vector = ClusteredState()
    e_prev = np.zeros(n_roots)
    for it in range(max_iter+1):
        print()
        print(" ===================================================================")
        print("     Selected CI Iteration: %4i epsilon: %12.8f" %(it,thresh_cipsi))
        print(" ===================================================================")
        print(" Build full Hamiltonian",flush=True)
        start = time.time()
        
        if it>0:
            H = grow_hamiltonian_parallel(H, clustered_ham, ci_vector, ci_vector_old)
        else:
            H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector, nproc=nproc)
        
        ci_vector_old = ci_vector.copy()
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
        v0 = v[:,:n_roots]
        e0 = e[:n_roots]

        old_dim = len(ci_vector)

        # store all vectors for the excited states
        all_vecs = []
        for rn in range(n_roots):
            print("Root:%4d     Energy:%12.8f  CI Dim: %4i "%(rn,e[rn].real,len(ci_vector)))
            vec = ci_vector.copy()
            vec.zero()
            vec.set_vector(v[:,rn])
            all_vecs.append(vec)
            vec.print()
        
        #check convergence
        e_diff = e0 - e_prev
        delta_e = np.linalg.norm(e_diff)
        e_prev = e0
        if abs(delta_e) < thresh_conv:
            print(" Converged: TPSCI")
            break
        print(" Next iteration CI space dimension", len(ci_vector))

        all_pt_vecs = []
        e2_energies = []
        for rn,vec in enumerate(all_vecs):
            start = time.time()
            if nbody_limit != 4:
                print(" Warning: nbody_limit set to %4i, resulting PT energies are meaningless" %nbody_limit)



            asci_vector = vec.copy()
            print(" Choose subspace from which to search for new configs. Thresh: ", thresh_asci)
            print(" CI Dim          : %8i" % len(asci_vector))
            kept_indices = asci_vector.clip(thresh_asci)
            print(" Search Dim      : %8i Norm: %12.8f" %( len(asci_vector), asci_vector.norm()))


            if matvec==1:
                pt_vector = matvec1_parallel1(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
            elif matvec==2:
                pt_vector = matvec1_parallel2(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
            elif matvec==3:
                pt_vector = matvec1_parallel3(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit)
            elif matvec==4:
                pt_vector = matvec1_parallel4(clustered_ham, asci_vector, nproc=nproc, thresh_search=thresh_search, nbody_limit=nbody_limit,
                        shared_mem=shared_mem, batch_size=batch_size)
            stop = time.time()
            print(" Time spent in matvec: %12.2f" %( stop-start))
            
            pt_vector.prune_empty_fock_spaces()


            print(" Remove CI space from pt_vector vector")
            for fockspace,configs in pt_vector.items():
                if fockspace in vec.fblocks():
                    for config,coeff in list(configs.items()):
                        if config in vec[fockspace]:
                            del pt_vector[fockspace][config]


            for fockspace,configs in vec.items():
                if fockspace in pt_vector:
                    for config,coeff in configs.items():
                        assert(config not in pt_vector[fockspace])

            print(" Norm of CI vector = %12.8f" %vec.norm())
            print(" Dimension of CI space: ", len(vec))
            print(" Dimension of PT space: ", len(pt_vector))
            if len(pt_vector) == 0:
                print("No more connecting config found")
                break
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
                print(" Time spent in denomimator: %12.2f" %( end - start), flush=True)
                
                denom = 1/(e0[rn] - Hd)
            elif pt_type == 'mp':
                start = time.time()
                # get barycentric MP zeroth order energy
                e0_mp = 0
                for f,c,v in vec:
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
                print(" Time spent in denomimator: %12.2f" %( end - start), flush=True)
            
            pt_vector_v = pt_vector.get_vector()
            pt_vector_v.shape = (pt_vector_v.shape[0])

            e2 = np.multiply(denom,pt_vector_v)
            pt_vector.set_vector(e2)
            e2 = np.dot(pt_vector_v,e2)

            print(" PT2 Energy Correction = %12.8f" %e2)
            print(" PT2 Energy Total      = %12.8f" %(e0[rn]+e2))

            all_pt_vecs.append(pt_vector)
            e2_energies.append(e2)


        if it >= max_iter:
            print(" Maxcycles: TPSCI")
            break

        for pt_vector in all_pt_vecs:
            print(" Choose which states to add to CI space", flush=True)
            for fockspace,configs in pt_vector.items():
                for config,coeff in configs.items():
                    if coeff*coeff > thresh_cipsi:
                        if fockspace in ci_vector:
                            ci_vector[fockspace][config] = 0
                        else:
                            ci_vector.add_fockspace(fockspace)
                            ci_vector[fockspace][config] = 0
            
            
            print(" Dimension of next CI space: ", len(ci_vector))

    return ci_vector, pt_vector, e0, e0+np.array(e2_energies)

# }}}


