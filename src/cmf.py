import math
import sys
import numpy as np
import scipy
import itertools
import copy as cp
from helpers import *
import opt_einsum as oe
import tools
import time

from ClusteredOperator import *
from ClusteredState import *
from Cluster import *

import ham_build
import matvec

def cmf(clustered_ham, ci_vector, h, g, 
            max_iter    = 20, 
            thresh      = 1e-8, 
            dm_guess    = None,
            diis        = False,
            diis_start  = 1,
            max_diis    = 6):
    """ Do CMF for a tensor product state 
       
        This modifies the data in clustered_ham.clusters, both the basis, and the operators such that the 
        cluster basis is only defined on the target fock space with only a single vector

   
        Input: 
        ci_vector   :   ClusteredState object which defines the fock space configuration for the CMF calcultion
                        Because CMF finds the best TPS wrt cluster state rotations, this should only have a single
                        configuration defined. If not, error will be thrown.

        h           :   1e integrals
        g           :   2e integrals

        thresh      :   how tightly to converge energy
        max_nroots  :   what is the max number of cluster states to allow in each fock space of each cluster
                            after the potential is optimized?
        dm_guess    :   Use initial guess for 1rdm? Typically one would use HF density matrix here. Input is a 
                        tuple of density matrices for alpha/beta: (Pa, Pb)

        Returns:
        energy      :   Final CMF energy
        converged   :   Bool indicating if it converged or not
        Da,Db       :   Tuple of optimized density matrices to be used for generating the rest of the cluster basis.
                        
    """
  # {{{
    print()
    print(" =================================================================")
    print("                         Do CMF")  
    print(" =================================================================")
    assert(len(ci_vector) == 1)
    ci_vector.prune_empty_fock_spaces()
    assert(len(ci_vector.fblocks())==1)
    for f in ci_vector.fblocks():
        fspace = f

    
    if dm_guess == None:
        rdm_a = np.zeros(h.shape)
        rdm_b = np.zeros(h.shape)
    else:
        rdm_a = dm_guess[0]
        rdm_b = dm_guess[1]
    
    assert(rdm_a.shape == h.shape)
    assert(rdm_b.shape == h.shape)
    converged = False
    clusters = clustered_ham.clusters
    e_last = 999

    if diis==True:
        diis_start = diis_start
        max_diis = max_diis
        diis_vals_dm = [rdm_a.copy()]
        diis_errors = []
        diis_size = 0

    ecore = clustered_ham.core_energy
    for cmf_iter in range(max_iter):
        print(" --------------------------------------")
        print("     Iteration:", cmf_iter)
        print(" --------------------------------------")

        rdm_a_old = rdm_a
        rdm_b_old = rdm_b
        
        print(" Build cluster basis and operators")
        for ci_idx, ci in enumerate(clusters):
            print(" ",ci)
            assert(ci_idx == ci.idx)
            ci.form_fockspace_eigbasis(h, g, [fspace[ci_idx]], 
                    max_roots=1, 
                    ecore=ecore,
                    rdm1_a=rdm_a, 
                    rdm1_b=rdm_b, 
                    iprint=1)
        
            print(" Build new operators for cluster ",ci.idx)
            ci.build_op_matrices(iprint=0)
            ci.build_local_terms(h,g)
      
        print(" Compute CMF energy")
        e_curr = ham_build.build_full_hamiltonian(clustered_ham, ci_vector)[0,0]

        print(" Converged?")
        if abs(e_curr-e_last) < thresh:
            print("*CMF Iter: %4i Energy: %20.12f Delta E: %12.1e" %(cmf_iter,e_curr, e_curr-e_last))
            print(" CMF Converged. ")
             
            # form 1rdm from reference state
            rdm_a, rdm_b = build_1rdm(ci_vector, clustered_ham.clusters)
            converged = True
            break
        elif abs(e_curr-e_last) >= thresh and cmf_iter == max_iter-1:
            print("?CMF Iter: %4i Energy: %20.12f Delta E: %12.1e" %(cmf_iter,e_curr, e_curr-e_last))
            print(" Max CMF iterations reached. Just continue anyway")
        elif abs(e_curr-e_last) >= thresh and cmf_iter < max_iter-1:
            print(" CMF Iter: %4i Energy: %20.12f Delta E: %12.1e" %(cmf_iter,e_curr, e_curr-e_last))
            print(" Continue CMF optimization")


            if diis==True:
                ###  DIIS  ###
                # form 1rdm from reference state
                old_dm = rdm_a.copy()
                rdm_a, rdm_b = build_1rdm(ci_vector, clustered_ham.clusters)
                dm_new = rdm_a.copy()

                diis_vals_dm.append(dm_new.copy())
                error_dm = (dm_new - old_dm).ravel()
                diis_errors.append(error_dm)

                if cmf_iter > diis_start:
                    # Limit size of DIIS vector
                    if (len(diis_vals_dm) > max_diis):
                        del diis_vals_dm[0]
                        del diis_errors[0]
                    diis_size = len(diis_vals_dm) - 1

                    # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
                    B = np.ones((diis_size + 1, diis_size + 1)) * -1
                    B[-1, -1] = 0

                    for n1, e1 in enumerate(diis_errors):
                        for n2, e2 in enumerate(diis_errors):
                            # Vectordot the error vectors
                            B[n1, n2] = np.dot(e1, e2)
                    B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()


                    # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
                    resid = np.zeros(diis_size + 1)
                    resid[-1] = -1

                    #print("B")
                    #print(B)
                    #print("resid")
                    #print(resid)
                    # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
                    ci = np.linalg.solve(B, resid)

                    # Calculate new amplitudes
                    dm_new[:] = 0

                    for num in range(diis_size):
                        dm_new += ci[num] * diis_vals_dm[num + 1]
                    # End DIIS amplitude update
                    
                    rdm_a = dm_new.copy()
                    rdm_b = dm_new.copy()
                    #print(rdm_a)
            elif diis==False:
                # form 1rdm from reference state
                rdm_a, rdm_b = build_1rdm(ci_vector, clustered_ham.clusters)

            e_last = e_curr
    
    
#    # Now compute the full basis and associated operators
#    print()
#    print(" CMF now done: Recompute the operator matrices in the new basis")
#    for ci_idx, ci in enumerate(clusters):
#    
#            
#        
#
#        # if delta_elec is set to None, build all possible fock spaces
#        if delta_elec == None:
#            max_e = ci.n_orb     
#            min_e = 0     
#        else:
#            nelec = fspace[ci_idx][0] + fspace[ci_idx][1]  
#            max_e = min(nelec+delta_elec, ci.n_orb)    
#            min_e = max(nelec-delta_elec, 0)
#
#        print()
#        print()
#        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
#        if rdm_a is not None and rdm_b is not None: 
#            ci.form_eigbasis_from_ints(h,g,max_roots=max_nroots, rdm1_a=rdm_a, rdm1_b=rdm_b, max_elec=max_e, min_nelec=min_e)
#        else:
#            ci.form_eigbasis_from_ints(h,g,max_roots=max_nroots, max_elec=max_e, min_nelec=min_e)
#        print(" Build these local operators")
#        print(" Build mats for cluster ",ci.idx)
#        ci.build_op_matrices()
#        ci.build_local_terms(h,g)

    return e_curr, converged, rdm_a, rdm_b
   # }}}

def build_1rdm(ci_vector, clusters):
    """
    Build 1rdm C_{I,J,K}<IJK|p'q|LMN> C_{L,M,N}
    """
    # {{{
    n_orb = ci_vector.n_orb
    dm_aa = np.zeros((n_orb,n_orb))
    dm_bb = np.zeros((n_orb,n_orb))
   
    if 0:   # doesn't work anymore after removing local terms from add_1b_terms
        print(ci_vector.norm())
        # build 1rdm in slow (easy) way
        dm_aa_slow = np.zeros((n_orb,n_orb))
        for i in range(n_orb):
            for j in range(n_orb):
                op = ClusteredOperator(clusters)
                h = np.zeros((n_orb,n_orb))
                h[i,j] = 1
                #op.add_local_terms()
                op.add_1b_terms(h)
                Nv = matvec.matvec1(op, ci_vector, thresh_search=1e-8)
                Nv = ci_vector.dot(Nv)
                dm_aa_slow[i,j] = Nv
        print(" Here is the slow version:")
        print(dm_aa_slow)
        occs = np.linalg.eig(dm_aa_slow)[0]
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        #with np.printoptions(precision=6, suppress=True):
        #    print(dm_aa_slow)
        print(" Trace slow:",np.trace(dm_aa_slow))
        exit()

    # define orbital index shifts
    tmp = 0
    shifts = []
    for ci in range(len(clusters)):
        shifts.append(tmp)
        tmp += clusters[ci].n_orb
   
    iprint = 1
 
    # Diagonal terms
    for fock in ci_vector.fblocks():
        for ci in clusters:
            #if ci.idx != 1:
            #    continue
            #Aa terms
            if fock[ci.idx][0] > 0:
                # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                # c(ijk...) <i|p'q|l> c(ljk...)
                for config_l in ci_vector.fblock(fock):
                    for config_r in ci_vector.fblock(fock):
                        # make sure all state indices are the same aside for clusters i and j
                        
                        delta_conf = [abs(config_l[i] - config_r[i]) for i in range(len(clusters))] 
                        delta_conf[ci.idx] = 0
                        
                        if sum(delta_conf) == 0:
                            pq = ci.get_op_mel('Aa', fock[ci.idx], fock[ci.idx], config_l[ci.idx], config_r[ci.idx])*ci_vector[fock][config_l] * ci_vector[fock][config_r]
                            dm_aa[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq
 
            #Bb terms
            if fock[ci.idx][1] > 0:
                # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                # c(ijk...) <i|p'q|l> c(ljk...)
                for config_l in ci_vector.fblock(fock):
                    for config_r in ci_vector.fblock(fock):
                        # make sure all state indices are the same aside for clusters i and j
                        delta_conf = [abs(config_l[i] - config_r[i]) for i in range(len(clusters))] 
                        delta_conf[ci.idx] = 0
                        
                        if sum(delta_conf) == 0:
                            pq = ci.get_op_mel('Bb', fock[ci.idx], fock[ci.idx], config_l[ci.idx], config_r[ci.idx])*ci_vector[fock][config_l] * ci_vector[fock][config_r]
                            dm_bb[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq
 
    # Off-diagonal terms
    for fock_l in ci_vector.fblocks():
        #continue 
        for ci in clusters:
            for cj in clusters:
                if cj.idx >= ci.idx:
                    continue
                #A,a terms
                if fock_l[cj.idx][0] < ci.n_orb and fock_l[ci.idx][0] > 0:
                    fock_r = list(fock_l)
                    fock_r[ci.idx] = tuple([fock_l[ci.idx][0]-1, fock_l[ci.idx][1]])
                    fock_r[cj.idx] = tuple([fock_l[cj.idx][0]+1, fock_l[cj.idx][1]])
                    fock_r = tuple(fock_r)
                    #print("A,a", fock_l, '-->', fock_r)
                    # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                    # c(ijk...) <i|p'|l> <j|q|m> c(lmk...) (-1)^N(l)
                    try:
                        for config_l in ci_vector.fblock(fock_l):
                            for config_r in ci_vector.fblock(fock_r):
                                # make sure all state indices are the same aside for clusters i and j
                                delta_conf = [abs(config_l[i]-config_r[i]) for i in range(len(clusters))] 
                                delta_conf[ci.idx] = 0
                                delta_conf[cj.idx] = 0
                                if sum(delta_conf) > 0:
                                    continue
                                #print(" Here:", config_l, config_r, delta_conf)
                                pmat = ci.get_op_mel('A', fock_l[ci.idx], fock_r[ci.idx], config_l[ci.idx], config_r[ci.idx])
                                qmat = cj.get_op_mel('a', fock_l[cj.idx], fock_r[cj.idx], config_l[cj.idx], config_r[cj.idx])
                                pq = np.einsum('p,q->pq',pmat,qmat) * ci_vector[fock_l][config_l] * ci_vector[fock_r][config_r]
                                pq.shape = (ci.n_orb,cj.n_orb)
                                # get state sign
                                state_sign = 1
                                for ck in range(ci.idx):
                                    state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                for ck in range(cj.idx):
                                    state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                pq = pq * state_sign
                                dm_aa[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[cj.idx]:shifts[cj.idx]+cj.n_orb] += pq
                                dm_aa[shifts[cj.idx]:shifts[cj.idx]+cj.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq.T
                    except KeyError:
                        pass 
                    
                
                #B,b terms
                if fock_l[cj.idx][1] < ci.n_orb and fock_l[ci.idx][1] > 0:
                    fock_r = list(fock_l)
                    fock_r[ci.idx] = tuple([fock_l[ci.idx][0], fock_l[ci.idx][1]-1])
                    fock_r[cj.idx] = tuple([fock_l[cj.idx][0], fock_l[cj.idx][1]+1])
                    fock_r = tuple(fock_r)
                    #print("A,a", fock_l, '-->', fock_r)
                    # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                    # c(ijk...) <i|p'|l> <j|q|m> c(lmk...) (-1)^N(l)
                    try:
                        for config_l in ci_vector.fblock(fock_l):
                            for config_r in ci_vector.fblock(fock_r):
                                # make sure all state indices are the same aside for clusters i and j
                                delta_conf = [abs(config_l[i]-config_r[i]) for i in range(len(clusters))] 
                                delta_conf[ci.idx] = 0
                                delta_conf[cj.idx] = 0
                                if sum(delta_conf) > 0:
                                    continue
                                #print(" Here:", config_l, config_r, delta_conf)
                                pmat = ci.get_op_mel('B', fock_l[ci.idx], fock_r[ci.idx], config_l[ci.idx], config_r[ci.idx])
                                qmat = cj.get_op_mel('b', fock_l[cj.idx], fock_r[cj.idx], config_l[cj.idx], config_r[cj.idx])
                                pq = np.einsum('p,q->pq',pmat,qmat) * ci_vector[fock_l][config_l] * ci_vector[fock_r][config_r]
                                pq.shape = (ci.n_orb,cj.n_orb)
                                # get state sign
                                state_sign = 1
                                for ck in range(ci.idx):
                                    state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                for ck in range(cj.idx):
                                    state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                pq = pq * state_sign
                                dm_bb[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[cj.idx]:shifts[cj.idx]+cj.n_orb] += pq
                                dm_bb[shifts[cj.idx]:shifts[cj.idx]+cj.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq.T
                    except KeyError:
                        pass 
                   

    # density is being made in a reindexed fasion - reorder now 
    new_index = []
    for ci in clusters:
        for cij in ci.orb_list:
            new_index.append(cij)
    new_index = np.array(new_index)

    idx = new_index.argsort()
    dm_aa = dm_aa[:,idx][idx,:]
    dm_bb = dm_bb[:,idx][idx,:]
                

    occs = np.linalg.eig(dm_aa + dm_bb)[0]
    print(" Eigenvalues of density matrix")
    [print(" %4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
    print(" Trace of Pa:", np.trace(dm_aa))
    print(" Trace of Pb:", np.trace(dm_bb))
    #with np.printoptions(precision=6, suppress=True):
    #    print(dm_aa + dm_bb)
    return dm_aa, dm_bb 

# }}}

def build_tdm(r_vector,l_vector,clustered_ham):
    """
    Build 1rdm C_{I,J,K}<IJK|p'q|LMN> C_{L,M,N}
    Build gradient for the CMF state. makes sure l_vector is only 1 config. 
    Computes Gpq =  <0|[H,p'q]|0>
                 =  <0|p'q.H|0> 
    We have H|0> =  matvec(l_vector)
    """
    # {{{
    n_orb = l_vector.n_orb
    dm_aa = np.zeros((n_orb,n_orb))
    dm_bb = np.zeros((n_orb,n_orb))
    clusters = clustered_ham.clusters

   
    if 0:   # doesn't work anymore after removing local terms from add_1b_terms
        print(l_vector.norm())
        # build 1rdm in slow (easy) way
        dm_aa_slow = np.zeros((n_orb,n_orb))
        for i in range(n_orb):
            for j in range(n_orb):
                op = ClusteredOperator(clusters)
                h = np.zeros((n_orb,n_orb))
                h[i,j] = 1
                #op.add_local_terms()
                op.add_1b_terms(h)
                Nv = matvec.matvec1(op, l_vector, thresh_search=1e-8)
                Nv = l_vector.dot(Nv)
                dm_aa_slow[i,j] = Nv
        print(" Here is the slow version:")
        print(dm_aa_slow)
        occs = np.linalg.eig(dm_aa_slow)[0]
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        #with np.printoptions(precision=6, suppress=True):
        #    print(dm_aa_slow)
        print(" Trace slow:",np.trace(dm_aa_slow))
        exit()

    # define orbital index shifts
    tmp = 0
    shifts = []
    for ci in range(len(clusters)):
        shifts.append(tmp)
        tmp += clusters[ci].n_orb
   
    iprint = 1
 
    # Diagonal terms
    for fock in l_vector.fblocks():
        for ci in clusters:
            #if ci.idx != 1:
            #    continue
            #Aa terms
            if fock[ci.idx][0] > 0:
                if fock in r_vector.fblocks():
                    # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                    # c(ijk...) <i|p'q|l> c(ljk...)
                    for config_l in l_vector.fblock(fock):
                        for config_r in r_vector.fblock(fock):
                            # make sure all state indices are the same aside for clusters i and j
                            
                            delta_conf = [abs(config_l[i] - config_r[i]) for i in range(len(clusters))] 
                            delta_conf[ci.idx] = 0
                            
                            if sum(delta_conf) == 0:
                                pq = ci.get_op_mel('Aa', fock[ci.idx], fock[ci.idx], config_l[ci.idx], config_r[ci.idx])*l_vector[fock][config_l] * r_vector[fock][config_r]
                                dm_aa[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq
 
            #Bb terms
            if fock[ci.idx][1] > 0:
                if fock in r_vector.fblocks():
                    # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                    # c(ijk...) <i|p'q|l> c(ljk...)
                    for config_l in l_vector.fblock(fock):
                        for config_r in r_vector.fblock(fock):
                            # make sure all state indices are the same aside for clusters i and j
                            delta_conf = [abs(config_l[i] - config_r[i]) for i in range(len(clusters))] 
                            delta_conf[ci.idx] = 0
                            
                            if sum(delta_conf) == 0:
                                pq = ci.get_op_mel('Bb', fock[ci.idx], fock[ci.idx], config_l[ci.idx], config_r[ci.idx])*l_vector[fock][config_l] * r_vector[fock][config_r]
                                dm_bb[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq
 
    # Off-diagonal terms
    for fock_l in l_vector.fblocks():
        #continue 
        for ci in clusters:
            for cj in clusters:
                if cj.idx >= ci.idx:
                    sign = -1
                else:
                    sign = 1
                #    continue
                #A,a terms
                if fock_l[cj.idx][0] < cj.n_orb and fock_l[ci.idx][0] > 0:
                    fock_r = list(fock_l)
                    fock_r[ci.idx] = tuple([fock_l[ci.idx][0]-1, fock_l[ci.idx][1]])
                    fock_r[cj.idx] = tuple([fock_l[cj.idx][0]+1, fock_l[cj.idx][1]])
                    fock_r = tuple(fock_r)

                    if fock_r in r_vector.fblocks():
                        #print("A,a", fock_l, '-->', fock_r)
                        # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                        # c(ijk...) <i|p'|l> <j|q|m> c(lmk...) (-1)^N(l)
                        try:
                            for config_l in l_vector.fblock(fock_l):
                                for config_r in r_vector.fblock(fock_r):
                                    # make sure all state indices are the same aside for clusters i and j
                                    delta_conf = [abs(config_l[i]-config_r[i]) for i in range(len(clusters))] 
                                    delta_conf[ci.idx] = 0
                                    delta_conf[cj.idx] = 0
                                    if sum(delta_conf) > 0:
                                        continue
                                    #print(" Here:", config_l, config_r, delta_conf)
                                    pmat = ci.get_op_mel('A', fock_l[ci.idx], fock_r[ci.idx], config_l[ci.idx], config_r[ci.idx])
                                    qmat = cj.get_op_mel('a', fock_l[cj.idx], fock_r[cj.idx], config_l[cj.idx], config_r[cj.idx])
                                    pq = np.einsum('p,q->pq',pmat,qmat) * l_vector[fock_l][config_l] * r_vector[fock_r][config_r]
                                    pq.shape = (ci.n_orb,cj.n_orb)
                                    # get state sign
                                    state_sign = 1
                                    for ck in range(ci.idx):
                                        state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                    for ck in range(cj.idx):
                                        state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                    pq = pq * state_sign
                                    dm_aa[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[cj.idx]:shifts[cj.idx]+cj.n_orb] += pq * sign
                                    #dm_aa[shifts[cj.idx]:shifts[cj.idx]+cj.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq.T
                        except KeyError:
                            pass 
                    
                
                #B,b terms
                if fock_l[cj.idx][1] < cj.n_orb and fock_l[ci.idx][1] > 0:
                    fock_r = list(fock_l)
                    fock_r[ci.idx] = tuple([fock_l[ci.idx][0], fock_l[ci.idx][1]-1])
                    fock_r[cj.idx] = tuple([fock_l[cj.idx][0], fock_l[cj.idx][1]+1])
                    fock_r = tuple(fock_r)

                    if fock_r in r_vector.fblocks():
                        #print("A,a", fock_l, '-->', fock_r)
                        # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                        # c(ijk...) <i|p'|l> <j|q|m> c(lmk...) (-1)^N(l)
                        try:
                            for config_l in l_vector.fblock(fock_l):
                                for config_r in r_vector.fblock(fock_r):
                                    # make sure all state indices are the same aside for clusters i and j
                                    delta_conf = [abs(config_l[i]-config_r[i]) for i in range(len(clusters))] 
                                    delta_conf[ci.idx] = 0
                                    delta_conf[cj.idx] = 0
                                    if sum(delta_conf) > 0:
                                        continue
                                    #print(" Here:", config_l, config_r, delta_conf)
                                    pmat = ci.get_op_mel('B', fock_l[ci.idx], fock_r[ci.idx], config_l[ci.idx], config_r[ci.idx])
                                    qmat = cj.get_op_mel('b', fock_l[cj.idx], fock_r[cj.idx], config_l[cj.idx], config_r[cj.idx])
                                    pq = np.einsum('p,q->pq',pmat,qmat) * l_vector[fock_l][config_l] * r_vector[fock_r][config_r]
                                    pq.shape = (ci.n_orb,cj.n_orb)
                                    # get state sign
                                    state_sign = 1
                                    for ck in range(ci.idx):
                                        state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                    for ck in range(cj.idx):
                                        state_sign *= (-1)**(fock_l[ck][0]+fock_l[ck][1])
                                    pq = pq * state_sign
                                    dm_bb[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[cj.idx]:shifts[cj.idx]+cj.n_orb] += pq * sign
                                    #dm_bb[shifts[cj.idx]:shifts[cj.idx]+cj.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq.T
                        except KeyError:
                            pass 
                       

    # density is being made in a reindexed fasion - reorder now 
    new_index = []
    for ci in clusters:
        for cij in ci.orb_list:
            new_index.append(cij)
    new_index = np.array(new_index)

    idx = new_index.argsort()
    dm_aa = dm_aa[:,idx][idx,:]
    dm_bb = dm_bb[:,idx][idx,:]
                

    occs = np.linalg.eig(dm_aa + dm_bb)[0]
    print(" Eigenvalues of density matrix")
    [print(" %4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
    print(" Trace of Pa:", np.trace(dm_aa))
    print(" Trace of Pb:", np.trace(dm_bb))
    #with np.printoptions(precision=6, suppress=True):
    #    print(dm_aa + dm_bb)
    return dm_aa, dm_bb 

# }}}


class CmfSolver:
    """

    """
    def __init__(self, h1, h2, ecore, blocks, init_fspace, C, 
            iprint          = 0,        # turn up for more verbose printing
            max_roots       = 1000,     # how many roots to use for matvec - should be full space for exactness
            matvec          = 1,        # which matvec function to use         
            do_intra_rots   = False,    # Should we allow orbital rotations inside a cluster?  
            cmf_ci_thresh   = 1e-10     # how tight to converge the coupled CAS problems?
            ):
        self.h = h1
        self.g = h2
        self.ecore = ecore
        self.C = C
        self.blocks = blocks
        self.init_fspace = init_fspace

        self.clustered_ham = None
        self.ci_vector = None
        self.e = 0
        self.cmf_dm_guess = None
        self.iter = 0
        self.gradient = None
        self.iprint = iprint 
        self.max_roots = max_roots  
        self.cmf_ci_thresh = cmf_ci_thresh # threshold for the inner loop
        self.matvec = matvec
        self.do_intra_rots = do_intra_rots
        self.to_freeze  = []

    def freeze_cluster_mixing(self,ci,cjlist):
        """
        prevent mixing orbitals between clusters ci and each cluster in cjlist
        A is either the gradient to hand back to the optimizer or the (square) antihermitian rotation matrix kappa
        """
        for cj in cjlist:
            self.to_freeze.append((ci,cj))
            self.to_freeze.append((cj,ci))

    
    def init(self,cmf_dm_guess=None):

        if cmf_dm_guess != None:
            self.cmf_dm_guess = cmf_dm_guess

        h = self.h
        g = self.g
        C = self.C
        blocks = self.blocks
        init_fspace = self.init_fspace
       
        if self.do_intra_rots == False:
            # freeze intra_block_rotations
            for bi in range(len(self.blocks)):
                self.freeze_cluster_mixing(bi,(bi,))

        n_blocks = len(blocks)
        clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
        
        print(" Clusters:")
        [print(ci) for ci in clusters]
        
        clustered_ham = ClusteredOperator(clusters, core_energy=self.ecore)
        print(" Add 1-body terms")
        clustered_ham.add_local_terms()
        clustered_ham.add_1b_terms(h)
        print(" Add 2-body terms")
        clustered_ham.add_2b_terms(g)

        ci_vector = ClusteredState()
        ci_vector.init(clusters, init_fspace)


        self.clustered_ham = clustered_ham
        self.ci_vector = ci_vector

        if self.cmf_dm_guess == None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter = 20, thresh = self.cmf_ci_thresh)

        if self.cmf_dm_guess != None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, 
                    diis        = True,
                    dm_guess    = self.cmf_dm_guess,
                    max_iter    = 200, 
                    thresh      = self.cmf_ci_thresh)

        # store rdm
        self.cmf_dm_guess = (rdm_a,rdm_b)

        # build cluster basis and operator matrices using CMF optimized density matrices
        for ci_idx, ci in enumerate(clustered_ham.clusters):
            #if delta_elec != None:
            #    fspaces_i = init_fspace[ci_idx]
            #    fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_elec) )
            #else:
            fspaces_i = ci.possible_fockspaces()
        
            print()
            print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
            ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=self.max_roots, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=self.ecore)
            
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)


    def energy_dps(self):
        edps = ham_build.build_hamiltonian_diagonal(self.clustered_ham,self.ci_vector)
        print("EDPS %16.8f"%edps)
        return edps

    def energy(self,Kpq):

        Kpq = Kpq.reshape(self.h.shape[0],self.h.shape[1])

        # remove frozen rotations
        for freeze in self.to_freeze:
            for bi in freeze:
                for bj in freeze:
                    for bii in self.blocks[bi]:
                        for bjj in self.blocks[bj]:
                            Kpq[bii,bjj] = 0

        if self.iprint > 0:
            print("Kpq")
            print(Kpq)

        h = self.h
        g = self.g
        C = self.C
        blocks = self.blocks
        init_fspace = self.init_fspace
        clustered_ham = self.clustered_ham
        ci_vector = self.ci_vector




        from scipy.sparse.linalg import expm
        U = expm(Kpq)
        C = C @ U
        #molden.from_mo(mol, 'h4.molden', C)
        h = U.T @ h @ U
        g = np.einsum("pqrs,pl->lqrs",g,U)
        g = np.einsum("lqrs,qm->lmrs",g,U)
        g = np.einsum("lmrs,rn->lmns",g,U)
        g = np.einsum("lmns,so->lmno",g,U)


        cmf_dm_guess =  (U.T @ self.cmf_dm_guess[0] @ U,U.T @ self.cmf_dm_guess[0] @ U)

        n_blocks = len(blocks)
        clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
        
        print(" Clusters:")
        [print(ci) for ci in clusters]
        
        clustered_ham = ClusteredOperator(clusters, core_energy=self.ecore)
        print(" Add 1-body terms")
        clustered_ham.add_local_terms()
        clustered_ham.add_1b_terms(h)
        print(" Add 2-body terms")
        clustered_ham.add_2b_terms(g)

        ci_vector = ClusteredState()
        ci_vector.init(clusters, init_fspace)



        if self.cmf_dm_guess == None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter = 20, thresh = self.cmf_ci_thresh)

        if self.cmf_dm_guess != None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, 
                    diis        = True,
                    dm_guess    = cmf_dm_guess,
                    max_iter    = 20, 
                    thresh      = self.cmf_ci_thresh)
        
        # store rdm (but first rotate them back to the reference basis
        self.cmf_dm_guess =  (U @ rdm_a @ U.T, U @ rdm_b @ U.T)

        print(" CMF In energy: %12.8f" %e_curr)
        # build cluster basis and operator matrices using CMF optimized density matrices
        for ci_idx, ci in enumerate(clusters):
            #if delta_elec != None:
            #    fspaces_i = init_fspace[ci_idx]
            #    fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_elec) )
            #else:
            fspaces_i = ci.possible_fockspaces()
        
            print()
            print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
            ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=self.max_roots, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=self.ecore)
            
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)

        self.e = e_curr

        return e_curr

    def callback(self, Kpq):
        print(" CMF Orbital Optimization: Iter:%4i Current Energy = %20.16f Gradient Norm %10.1e Gradient Max %10.1e" %(self.iter,
                self.e, np.linalg.norm(self.gradient), np.max(np.abs(self.gradient))))
        self.iter += 1
        sys.stdout.flush()

    def rotate(self,Kpq):
        
        Kpq = Kpq.reshape(self.h.shape[0],self.h.shape[1])
        # remove frozen rotations
        for freeze in self.to_freeze:
            for bi in freeze:
                for bj in freeze:
                    for bii in self.blocks[bi]:
                        for bjj in self.blocks[bj]:
                            Kpq[bii,bjj] = 0

        h = self.h
        g = self.g
        C = self.C
        blocks = self.blocks
        init_fspace = self.init_fspace
        clustered_ham = self.clustered_ham
        ci_vector = self.ci_vector
        cmf_dm_guess = self.cmf_dm_guess


        from scipy.sparse.linalg import expm
        U = expm(Kpq) #form unitary

        C = C @ U  #rotate coeff
        h = U.T @ h @ U
        print(h)
        g = np.einsum("pqrs,pl->lqrs",g,U)
        g = np.einsum("lqrs,qm->lmrs",g,U)
        g = np.einsum("lmrs,rn->lmns",g,U)
        g = np.einsum("lmns,so->lmno",g,U)

        cmf_dm_guess =  (U.T @ self.cmf_dm_guess[0] @ U,U.T @ self.cmf_dm_guess[0] @ U)

        n_blocks = len(blocks)
        clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
        
        print(" Clusters:")
        [print(ci) for ci in clusters]
        
        clustered_ham = ClusteredOperator(clusters, core_energy=self.ecore)
        print(" Add 1-body terms")
        clustered_ham.add_local_terms()
        clustered_ham.add_1b_terms(h)
        print(" Add 2-body terms")
        clustered_ham.add_2b_terms(g)

        ci_vector = ClusteredState()
        ci_vector.init(clusters, init_fspace)

        if self.cmf_dm_guess == None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter = 20, thresh = self.cmf_ci_thresh)

        if self.cmf_dm_guess != None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, 
                    diis        = True,
                    dm_guess    = self.cmf_dm_guess,
                    max_iter    = 20, 
                    thresh      = self.cmf_ci_thresh)

        # build cluster basis and operator matrices using CMF optimized density matrices
        for ci_idx, ci in enumerate(clustered_ham.clusters):
            #if delta_elec != None:
            #    fspaces_i = init_fspace[ci_idx]
            #    fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_elec) )
            #else:
            fspaces_i = ci.possible_fockspaces()
        
            print()
            print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
            ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=self.max_roots, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=self.ecore)
            
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)


        self.h = h
        self.g = g
        self.C = C
        self.clustered_ham = clustered_ham
        self.ci_vector = ci_vector
        self.cmf_dm_guess = cmf_dm_guess


    def grad(self,Kpq):

        Kpq = Kpq.reshape(self.h.shape[0],self.h.shape[1])
        # remove frozen rotations
        print(self.to_freeze)
        for freeze in self.to_freeze:
            for bi in freeze:
                for bj in freeze:
                    print(" Freeze orbital mixing between clusters %4i and %4i"%(bi,bj)) 
                    for bii in self.blocks[bi]:
                        for bjj in self.blocks[bj]:
                            Kpq[bii,bjj] = 0

        h = self.h
        g = self.g
        C = self.C
        blocks = self.blocks
        init_fspace = self.init_fspace
        clustered_ham = self.clustered_ham
        ci_vector = self.ci_vector


        from scipy.sparse.linalg import expm
        U = expm(Kpq)
        print(U)
        print(U.T @ U)
        print(h)
        C = C @ U
        #molden.from_mo(mol, 'h4.molden', C)
        h = U.T @ h @ U
        print(h)
        g = np.einsum("pqrs,pl->lqrs",g,U)
        g = np.einsum("lqrs,qm->lmrs",g,U)
        g = np.einsum("lmrs,rn->lmns",g,U)
        g = np.einsum("lmns,so->lmno",g,U)


        cmf_dm_guess =  (U.T @ self.cmf_dm_guess[0] @ U,U.T @ self.cmf_dm_guess[0] @ U)

        n_blocks = len(blocks)
        clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
        
        print(" Clusters:")
        [print(ci) for ci in clusters]
        
        clustered_ham = ClusteredOperator(clusters, core_energy=self.ecore)
        print(" Add 1-body terms")
        clustered_ham.add_local_terms()
        clustered_ham.add_1b_terms(h)
        print(" Add 2-body terms")
        clustered_ham.add_2b_terms(g)

        ci_vector = ClusteredState()
        ci_vector.init(clusters, init_fspace)



        if self.cmf_dm_guess == None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter = 20, thresh = self.cmf_ci_thresh)

        if self.cmf_dm_guess != None:
            e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, 
                    diis        = True,
                    dm_guess    = self.cmf_dm_guess,
                    max_iter    = 20, 
                    thresh      = self.cmf_ci_thresh)
        
        # store rdm (but first rotate them back to the reference basis
        self.cmf_dm_guess =  (U @ rdm_a @ U.T, U @ rdm_b @ U.T)

        
        print(" CMF In grad  : %12.8f" %e_curr)
        # build cluster basis and operator matrices using CMF optimized density matrices
        for ci_idx, ci in enumerate(clusters):
            #if delta_elec != None:
            #    fspaces_i = init_fspace[ci_idx]
            #    fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_elec) )
            #else:
            fspaces_i = ci.possible_fockspaces()
        
            print()
            print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
            ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=self.max_roots, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=self.ecore)
            
            print(" Build operator matrices for cluster ",ci.idx)
            ci.build_op_matrices()
            ci.build_local_terms(h,g)




        if self.matvec==1:
            h1_vector = matvec.matvec1(clustered_ham, ci_vector, thresh_search=0, nbody_limit=3)
        elif self.matvec==2:
            pt_vector = matvec.matvec1_parallel2(clustered_ham, ci_vector, nproc=self.nproc, nbody_limit=3)
        elif self.matvec==3:
            pt_vector = matvec.matvec1_parallel3(clustered_ham, ci_vector, nproc=self.nproc, nbody_limit=3)
        #elif matvec==4:
        #    pt_vector = matvec.matvec1_parallel4(clustered_ham, ci_vector, nproc=self.nproc, nbody_limit=3,
        #            shared_mem=shared_mem, batch_size=batch_size)
        else:
            print(" Wrong option for matvec")
            exit()
        #h1_vector.print_configs()
        rdm_a1, rdm_b1 = build_tdm(ci_vector,h1_vector,clustered_ham)
        rdm_a2, rdm_b2 = build_tdm(h1_vector,ci_vector,clustered_ham)
        print("Gradient")
        Gpq = rdm_a1+rdm_b1-rdm_a2-rdm_b2
        print(Gpq)
        print("NormGrad1",np.linalg.norm(Gpq))

        #print("CurrCMF:%12.8f       Grad:%12.8f    dE:%10.6f"%(e_curr,np.linalg.norm(grad),e_curr-self.e))
        #print("CurrCMF:%12.8f       Grad:%12.8f   "%(e_curr,np.linalg.norm(Gpq)))
        

        # remove frozen rotations
        for freeze in self.to_freeze:
            for bi in freeze:
                for bj in freeze:
                    for bii in self.blocks[bi]:
                        for bjj in self.blocks[bj]:
                            Gpq[bii,bjj] = 0

        self.gradient = Gpq.ravel()
        return Gpq.ravel()

