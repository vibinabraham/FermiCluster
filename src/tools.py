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


def cmf(clustered_ham, ci_vector, h, g, max_iter=20, thresh=1e-8, max_nroots=1000,dm_guess=None,diis=False,diis_start=1,max_diis=6):
    """ Do CMF for a tensor product state 
       
        This modifies the data in clustered_ham.clusters, both the basis, and the operators

        h is the 1e integrals
        g is the 2e integrals

        thresh: how tightly to converge energy
        max_nroots: what is the max number of cluster states to allow in each fock space of each cluster
                    after the potential is optimized?
    """
  # {{{


    if dm_guess == None:
        rdm_a = None
        rdm_b = None
    else:
        rdm_a = dm_guess[0]
        rdm_b = dm_guess[1]
    converged = False
    clusters = clustered_ham.clusters
    e_last = 999

    if diis==True:
        diis_start = diis_start
        max_diis = max_diis
        diis_vals_dm = [rdm_a.copy()]
        diis_errors = []
        diis_size = 0

    for cmf_iter in range(max_iter):
        rdm_a_old = rdm_a
        rdm_b_old = rdm_b
        
        print(" Build cluster basis and operators")
        for ci_idx, ci in enumerate(clusters):
            assert(ci_idx == ci.idx)
            if cmf_iter > 0:
                ci.form_eigbasis_from_ints(h,g,max_roots=1, rdm1_a=rdm_a, rdm1_b=rdm_b)
            elif dm_guess != None: 
                ci.form_eigbasis_from_ints(h,g,max_roots=1, rdm1_a=rdm_a, rdm1_b=rdm_b)
            else:
                ci.form_eigbasis_from_ints(h,g,max_roots=1)
        
            print(" Build new operators for cluster ",ci.idx)
            ci.build_op_matrices()
      
        print(" Compute CMF energy")
        e_curr = build_full_hamiltonian(clustered_ham, ci_vector)[0,0]
        print(" CMF Iter: %4i Energy: %12.8f" %(cmf_iter,e_curr))

        print(" Converged?")
        if abs(e_curr-e_last) < thresh:
            print(" CMF Converged. ")
             
            # form 1rdm from reference state
            rdm_a, rdm_b = tools.build_1rdm(ci_vector)
            converged = True
            break
        elif abs(e_curr-e_last) >= thresh and cmf_iter == max_iter-1:
            print(" Max CMF iterations reached. Just continue anyway")
        elif abs(e_curr-e_last) >= thresh and cmf_iter < max_iter-1:
            print(" Continue CMF optimization")


            if diis==True:
                ###  DIIS  ###
                # form 1rdm from reference state
                old_dm = rdm_a.copy()
                rdm_a, rdm_b = tools.build_1rdm(ci_vector)
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

                    print("B")
                    print(B)
                    print("resid")
                    print(resid)
                    # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
                    ci = np.linalg.solve(B, resid)

                    # Calculate new amplitudes
                    dm_new[:] = 0

                    for num in range(diis_size):
                        dm_new += ci[num] * diis_vals_dm[num + 1]
                    # End DIIS amplitude update
                    
                    rdm_a = dm_new.copy()
                    rdm_b = dm_new.copy()
                    print(rdm_a)
            elif diis==False:
                # form 1rdm from reference state
                rdm_a, rdm_b = tools.build_1rdm(ci_vector)

            e_last = e_curr
    
    
    # Now compute the full basis and associated operators
    for ci_idx, ci in enumerate(clusters):
        print()
        print()
        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
        if rdm_a is not None and rdm_b is not None: 
            ci.form_eigbasis_from_ints(h,g,max_roots=max_nroots, rdm1_a=rdm_a, rdm1_b=rdm_b)
        else:
            ci.form_eigbasis_from_ints(h,g,max_roots=max_nroots)
        print(" Build these local operators")
        print(" Build mats for cluster ",ci.idx)
        ci.build_op_matrices()

    return e_curr,converged
   # }}}

    
def matvec1(h,v,term_thresh=1e-12):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 

    """
# {{{
    clusters = h.clusters
    sigma = ClusteredState(clusters)
    sigma = v.copy() 
    sigma.zero()
   
    if 0:
        # use this to debug
        sigma.expand_to_full_space()

    # loop over fock space blocks in the input ClusteredState vector, r/l means right/left hand side 
    for fock_ri, fock_r in enumerate(v.fblocks()):

        for terms in h.terms:
            # each item in terms will transition from the current fock config (fock_r) to a new fock config, fock_l 
            fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(clusters))])

            # check to make sure this fock config doesn't have negative or too many electrons in any cluster
            good = True
            for c in clusters:
                if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                    good = False
                    break
            if good == False:
                continue
            
            
            if fock_l not in sigma.data:
                sigma.add_fockspace(fock_l)

            configs_l = sigma[fock_l] 
           
            for term in h.terms[terms]:
                #print(" term: ", term)
                state_sign = 1
                for oi,o in enumerate(term.ops):
                    if o == '':
                        continue
                    if len(o) == 1 or len(o) == 3:
                        for cj in range(oi):
                            state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                    
                for conf_ri, conf_r in enumerate(v[fock_r]):
                    #print("  ", conf_r)
                    
                    #if abs(v[fock_r][conf_r]) < 5e-2:
                    #    continue
                    # get state sign 
                    #print('state_sign ', state_sign)
                    opii = -1
                    mats = []
                    good = True
                    for opi,op in enumerate(term.ops):
                        if op == "":
                            continue
                        opii += 1
                        #print(opi,term.active)
                        ci = clusters[opi]
                        #ci = clusters[term.active[opii]]
                        try:
                            oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                            mats.append(oi)
                        except KeyError:
                            good = False
                            break
                    if good == False:
                        continue                        
                        #break
                   
                    if len(mats) == 0:
                        continue
                    #print('mats:', end='')
                    #[print(m.shape,end='') for m in mats]
                    #print()
                    #print('ints:', term.ints.shape)
                    #print("contract_string       :", term.contract_string)
                    #print("contract_string_matvec:", term.contract_string_matvec)
                    
                    
                    #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints)
                    

                    v_coeff = v[fock_r][conf_r]
                    tmp = state_sign * tmp.ravel() * v_coeff

                    new_configs = [[i] for i in conf_r] 
                    for cacti,cact in enumerate(term.active):
                        new_configs[cact] = range(mats[cacti].shape[0])
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
                        if abs(tmp[sp_idx]) > term_thresh:
                            if spi not in configs_l:
                                configs_l[spi] = tmp[sp_idx] 
                            else:
                                configs_l[spi] += tmp[sp_idx] 
    
    print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    return sigma 
# }}}


def matvec1_parallel1(h_in,v,term_thresh=1e-12, nproc=None):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 
    """
# {{{
    global h 
    global clusters
    global sigma 
    print(" In matvec1_parallel1. nproc=",nproc) 
    h = h_in
    clusters = h_in.clusters
    

    sigma = ClusteredState(clusters)
    sigma = v.copy() 
    sigma.zero()
   
    if 0:
        # use this to debug
        sigma.expand_to_full_space()


    def do_parallel_work(v_curr):
        fock_r = v_curr[0]
        conf_r = v_curr[1]
        coeff  = v_curr[2]
        
        #sigma_out = ClusteredState(clusters)
        sigma_out = OrderedDict() 
        for terms in h.terms:
            fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(clusters))])
            good = True
            for c in clusters:
                if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                    good = False
                    break
            if good == False:
                continue
            
            #print(fock_l, "<--", fock_r)
            
            #if fock_l not in sigma_out.data:
            if fock_l not in sigma_out:
                sigma_out[fock_l] = OrderedDict()
            
            configs_l = sigma_out[fock_l] 
            
            for term in h.terms[terms]:
                #print(" term: ", term)
                state_sign = 1
                for oi,o in enumerate(term.ops):
                    if o == '':
                        continue
                    if len(o) == 1 or len(o) == 3:
                        for cj in range(oi):
                            state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                    
                #print("  ", conf_r)
                
                #if abs(v[fock_r][conf_r]) < 5e-2:
                #    continue
                # get state sign 
                #print('state_sign ', state_sign)
                opii = -1
                mats = []
                good = True
                for opi,op in enumerate(term.ops):
                    if op == "":
                        continue
                    opii += 1
                    #print(opi,term.active)
                    ci = clusters[opi]
                    #ci = clusters[term.active[opii]]
                    try:
                        oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                        mats.append(oi)
                    except KeyError:
                        good = False
                        break
                if good == False:
                    continue                        
                    #break
               
                if len(mats) == 0:
                    continue
                #print('mats:', end='')
                #[print(m.shape,end='') for m in mats]
                #print()
                #print('ints:', term.ints.shape)
                #print("contract_string       :", term.contract_string)
                #print("contract_string_matvec:", term.contract_string_matvec)
                
                
                #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                tmp = np.einsum(term.contract_string_matvec, *mats, term.ints)
                
        
                #v_coeff = v[fock_r][conf_r]
                #tmp = state_sign * tmp.ravel() * v_coeff
                tmp = state_sign * tmp.ravel() * coeff 
        
                new_configs = [[i] for i in conf_r] 
                for cacti,cact in enumerate(term.active):
                    new_configs[cact] = range(mats[cacti].shape[0])
                for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                    #print(" New config: %12.8f" %tmp[sp_idx], spi)
                    if abs(tmp[sp_idx]) > term_thresh:
                        if spi not in configs_l:
                            configs_l[spi] = tmp[sp_idx] 
                        else:
                            configs_l[spi] += tmp[sp_idx] 
        return sigma_out
    
    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)

    out = pool.map(do_parallel_work, v, batches=100)
    pool.close()
    pool.join()
    pool.clear()
    #out = list(map(do_parallel_work, v))
    print(" This is how much memory is being used to store matvec results:    ",sys.getsizeof(out)) 
    for o in out:
        sigma.add(o)

    print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    return sigma 
# }}}


def matvec1_parallel2(h_in,v,term_thresh=1e-12, nproc=None):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 
    """
# {{{
    global h 
    global clusters
    global sigma 
    print(" In matvec1_parallel1. nproc=",nproc) 
    h = h_in
    clusters = h_in.clusters
    

    sigma = ClusteredState(clusters)
    sigma = v.copy() 
    sigma.zero()
    

    def do_batch(batch):
        sigma_out = OrderedDict() 
        for v_curr in batch:
            do_parallel_work(v_curr,sigma_out)
        return sigma_out

    def do_parallel_work(v_curr,sigma_out):
        fock_r = v_curr[0]
        conf_r = v_curr[1]
        coeff  = v_curr[2]
        
        #sigma_out = ClusteredState(clusters)
        for terms in h.terms:
            fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(clusters))])
            good = True
            for c in clusters:
                if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                    good = False
                    break
            if good == False:
                continue
            
            #print(fock_l, "<--", fock_r)
            
            #if fock_l not in sigma_out.data:
            if fock_l not in sigma_out:
                sigma_out[fock_l] = OrderedDict()
            
            configs_l = sigma_out[fock_l] 
            
            for term in h.terms[terms]:
                #print(" term: ", term)
                state_sign = 1
                for oi,o in enumerate(term.ops):
                    if o == '':
                        continue
                    if len(o) == 1 or len(o) == 3:
                        for cj in range(oi):
                            state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                    
                #print("  ", conf_r)
                
                #if abs(v[fock_r][conf_r]) < 5e-2:
                #    continue
                # get state sign 
                #print('state_sign ', state_sign)
                opii = -1
                mats = []
                good = True
                for opi,op in enumerate(term.ops):
                    if op == "":
                        continue
                    opii += 1
                    #print(opi,term.active)
                    ci = clusters[opi]
                    #ci = clusters[term.active[opii]]
                    try:
                        oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                        mats.append(oi)
                    except KeyError:
                        good = False
                        break
                if good == False:
                    continue                        
                    #break
               
                if len(mats) == 0:
                    continue
                #print('mats:', end='')
                #[print(m.shape,end='') for m in mats]
                #print()
                #print('ints:', term.ints.shape)
                #print("contract_string       :", term.contract_string)
                #print("contract_string_matvec:", term.contract_string_matvec)
                
                
                #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                tmp = np.einsum(term.contract_string_matvec, *mats, term.ints)
                
        
                #v_coeff = v[fock_r][conf_r]
                #tmp = state_sign * tmp.ravel() * v_coeff
                tmp = state_sign * tmp.ravel() * coeff 
        
                new_configs = [[i] for i in conf_r] 
                for cacti,cact in enumerate(term.active):
                    new_configs[cact] = range(mats[cacti].shape[0])
                for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                    #print(" New config: %12.8f" %tmp[sp_idx], spi)
                    if abs(tmp[sp_idx]) > term_thresh:
                        if spi not in configs_l:
                            configs_l[spi] = tmp[sp_idx] 
                        else:
                            configs_l[spi] += tmp[sp_idx] 
        return sigma_out
    
    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)
   
   

    print(" Using Pathos library for parallelization. Number of workers: ", pool.ncpus, flush=True )
    # define batches
    conf_batches = []
    batch_size = math.ceil(len(v)/pool.ncpus)
    batch = []
    print(" Form batches. Max batch size: ", batch_size)
    for i,j,k in v:
        if len(batch) < batch_size:
            batch.append((i,j,k))
        else:
            conf_batches.append(batch)
            batch = []
            batch.append((i,j,k))
    if len(batch) > 0:
        conf_batches.append(batch)
        batch = []

    tmp = 0
    for b in conf_batches:
        for bi in b:
            tmp += 1
    assert(len(v) == tmp)


    out = pool.map(do_batch, conf_batches)
    #out = pool.map(do_parallel_work, v, batches=100)
    pool.close()
    pool.join()
    pool.clear()
    #out = list(map(do_parallel_work, v))
    print(" This is how much memory is being used to store matvec results:    ",sys.getsizeof(out)) 
    for o in out:
        sigma.add(o)

    print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    return sigma 
# }}}


def heat_bath_search(h_in,v,thresh_cipsi=None, nproc=None):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 
    """
# {{{
    global h 
    global clusters
    global sigma 
    print(" In heat_batch_search. nproc=",nproc) 
    h = h_in
    clusters = h_in.clusters
    

    sigma = ClusteredState(clusters)
    sigma = v.copy() 
    sigma.zero()
    
    sqrt_thresh_cipsi = np.sqrt(thresh_cipsi)

    def do_batch(batch):
        sigma_out = OrderedDict() 
        for v_curr in batch:
            do_parallel_work(v_curr,sigma_out)
        return sigma_out

    def do_parallel_work(v_curr,sigma_out):
        fock_r = v_curr[0]
        conf_r = v_curr[1]
        coeff  = v_curr[2]
        
        #sigma_out = ClusteredState(clusters)
        for terms in h.terms:
            fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(clusters))])
            good = True
            for c in clusters:
                if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                    good = False
                    break
            if good == False:
                continue
            
            #print(fock_l, "<--", fock_r)
            
            #if fock_l not in sigma_out.data:
            if fock_l not in sigma_out:
                sigma_out[fock_l] = OrderedDict()
            
            configs_l = sigma_out[fock_l] 
            
            for term in h.terms[terms]:
                #print(" term: ", term)
                state_sign = 1
                for oi,o in enumerate(term.ops):
                    if o == '':
                        continue
                    if len(o) == 1 or len(o) == 3:
                        for cj in range(oi):
                            state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                    
                #print("  ", conf_r)
                
                #if abs(v[fock_r][conf_r]) < 5e-2:
                #    continue
                # get state sign 
                #print('state_sign ', state_sign)
                opii = -1
                mats = []
                good = True
                for opi,op in enumerate(term.ops):
                    if op == "":
                        continue
                    opii += 1
                    #print(opi,term.active)
                    ci = clusters[opi]
                    #ci = clusters[term.active[opii]]
                    try:
                        oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                        mats.append(oi)
                    except KeyError:
                        good = False
                        break
                if good == False:
                    continue                        
                    #break
               
                if len(mats) == 0:
                    continue
                #print('mats:', end='')
                #[print(m.shape,end='') for m in mats]
                #print()
                #print('ints:', term.ints.shape)
                #print("contract_string       :", term.contract_string)
                #print("contract_string_matvec:", term.contract_string_matvec)
                
                
                #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                tmp = np.einsum(term.contract_string_matvec, *mats, term.ints)
        
                #v_coeff = v[fock_r][conf_r]
                #tmp = state_sign * tmp.ravel() * v_coeff
                #tmp = np.abs(state_sign * tmp.ravel() * coeff )
                tmp = state_sign * tmp.ravel() * coeff 
        
                new_configs = [[i] for i in conf_r] 
                for cacti,cact in enumerate(term.active):
                    new_configs[cact] = range(mats[cacti].shape[0])
                for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                    #print(" New config: %12.8f" %tmp[sp_idx], spi)
                    #if abs(tmp[sp_idx]) > sqrt_thresh_cipsi/10.0:
                    if abs(tmp[sp_idx]) > sqrt_thresh_cipsi:
                        #configs_l[spi] = tmp[sp_idx] # since we are only finding configs, we don't care about coeff
                        #configs_tmp.append([configs_l[spi],tmp[sp_idx]]) 
                        if spi not in configs_l:
                            configs_l[spi] = tmp[sp_idx] 
                        else:
                            configs_l[spi] += tmp[sp_idx] 

            #for config,coeff in list(configs_l.items()):
            #    if abs(coeff) < sqrt_thresh_cipsi:
            #        del configs_l[config]
        for fockspace,configs in sigma_out.items():
            for config,coeff in list(configs.items()):
                if abs(coeff) < sqrt_thresh_cipsi:
                    del sigma_out[fockspace][config]
        return sigma_out
    
    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)
   
   

    print(" Using Pathos library for parallelization. Number of workers: ", pool.ncpus, flush=True )
    # define batches
    conf_batches = []
    batch_size = math.ceil(len(v)/pool.ncpus)
    batch = []
    print(" Form batches. Max batch size: ", batch_size)
    for i,j,k in v:
        if len(batch) < batch_size:
            batch.append((i,j,k))
        else:
            conf_batches.append(batch)
            batch = []
            batch.append((i,j,k))
    if len(batch) > 0:
        conf_batches.append(batch)
        batch = []

    tmp = 0
    for b in conf_batches:
        for bi in b:
            tmp += 1
    assert(len(v) == tmp)


    out = pool.map(do_batch, conf_batches)
    #out = pool.map(do_parallel_work, v, batches=100)
    pool.close()
    pool.join()
    pool.clear()
    #out = list(map(do_parallel_work, v))
    print(" This is how much memory is being used to store matvec results:    ",sys.getsizeof(out)) 
    for o in out:
        sigma.add(o)

    # I'm not sure if we want to do this, since it's exactly the same importance function 
    sigma.clip(sqrt_thresh_cipsi)

    print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    return sigma 
# }}}


def build_full_hamiltonian(clustered_ham,ci_vector,iprint=0):
    """
    Build hamiltonian in basis in ci_vector
    """
# {{{
    clusters = ci_vector.clusters
    H = np.zeros((len(ci_vector),len(ci_vector)))
    
    shift_l = 0 
    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        if iprint > 0:
            print(fock_l)
       
        for config_li, config_l in enumerate(configs_l):
            idx_l = shift_l + config_li 
            
            shift_r = 0 
            for fock_ri, fock_r in enumerate(ci_vector.data):
                configs_r = ci_vector[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                if fock_ri<fock_li:
                    shift_r += len(configs_r) 
                    continue
                try:
                    terms = clustered_ham.terms[delta_fock]
                except KeyError:
                    shift_r += len(configs_r) 
                    continue 
                
                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = shift_r + config_ri
                    if idx_r<idx_l:
                        continue
                    
                    for term in terms:
                        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        H[idx_l,idx_r] += me
                        if idx_r>idx_l:
                            H[idx_r,idx_l] += me
                        #print(" %4i %4i = %12.8f"%(idx_l,idx_r,me),"  :  ",config_l,config_r, " :: ", term)
                shift_r += len(configs_r) 
        shift_l += len(configs_l)
    return H
# }}}


def build_full_hamiltonian_open(clustered_ham,ci_vector,iprint=1):
    """
    Build hamiltonian in basis in ci_vector
    """
# {{{
    clusters = ci_vector.clusters
    H = np.zeros((len(ci_vector),len(ci_vector)))
    n_clusters = len(clusters)

    fock_space_shifts = [0]
    for fi,f in enumerate(ci_vector.fblocks()):
        configs_i = ci_vector[f]
        fock_space_shifts.append(fock_space_shifts[-1]+len(configs_i))


    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        for fock_ri, fock_r in enumerate(ci_vector.data):
            if fock_li > fock_ri:
                continue
            configs_r = ci_vector[fock_r]

            delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
            
           


            try:
                terms = clustered_ham.terms[delta_fock]
            except KeyError:
                continue 
            for term in terms:
                # Compute the state sign now - since it only depends on fock spaces
                state_sign = 1

                term_exists = True
                for oi,o in enumerate(term.ops):
                    if o == '':
                        continue
                    if len(o) == 1 or len(o) == 3:
                        for cj in range(oi):
                            state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                
                    # Check to make sure each cluster is allowed to make the requested transition
                    try:
                        do = clusters[oi].ops[o]
                    except:
                        print(" Couldn't find:", term)
                        exit()
                    try:
                        d = do[(fock_l[oi],fock_r[oi])]
                        #d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
                    except:
                        term_exists = False
                if not term_exists:
                    continue 

                
                for config_li, config_l in enumerate(configs_l):
                    idx_l = fock_space_shifts[fock_li] + config_li 
                    for config_ri, config_r in enumerate(configs_r):        
                        idx_r = fock_space_shifts[fock_ri] + config_ri 
                        
                        if idx_r<idx_l:
                            continue
                
                        # Check to make sure each cluster is diagonal except if active
                        allowed = True
                        for ci in range(n_clusters):
                            if (config_l[ci]!=config_r[ci]) and (ci not in term.active):
                                allowed = False
                        if not allowed:
                            continue
                       

                        #d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
                        mats = []
                        for ci in term.active:
                            mats.append( clusters[ci].ops[term.ops[ci]][(fock_l[ci],fock_r[ci])][config_l[ci],config_r[ci]] ) 

                        me = 0.0
                      
                        if len(mats) != len(term.active):
                            continue
                        
                        #check that the mats where treated as views and also contiguous
                        #for m in mats:
                        #    print(m.flags['OWNDATA'])  #False -- apparently this is a view
                        #    print(m.__array_interface__)
                        #    print()

                        # todo:
                        #    For some reason, precompiled contract expression is slower than direct einsum - figure this out
                        #me = term.contract_expression(*mats) * state_sign
                        me = np.einsum(term.contract_string,*mats,term.ints) * state_sign

#                        me2 = term.matrix_element(fock_l,config_l,fock_r,config_r)
#                        try:
#                            assert(abs(me - me2) < 1e-8)
#                        except:
#                            print(term)
#                            print(mats)
#                            print(me)
#                            print(me2)
#                            exit()
                        H[idx_l,idx_r] += me
                        if idx_r>idx_l:
                            H[idx_r,idx_l] += me
    return H
    

# }}}


def build_full_hamiltonian_parallel1(clustered_ham_in,ci_vector_in,iprint=1, nproc=None):
    """
    Build hamiltonian in basis in ci_vector
    """
# {{{
    global clusters
    global ci_vector
    global clustered_ham
    
    print(" In build_full_hamiltonian_parallel1. nproc=",nproc) 

    clustered_ham = clustered_ham_in
    ci_vector = ci_vector_in
    clusters = ci_vector.clusters

    H = np.zeros((len(ci_vector),len(ci_vector)))
    n_clusters = len(clusters)

    def compute_parallel_block(f):
        fock_l  = f[0]
        fock_li = f[1]
        fock_r  = f[2]
        fock_ri = f[3]
    
        diagonal = False
        if fock_l == fock_r:
            diagonal = True

        if fock_li > fock_ri:
            return 
        
        #print("Processing the block: ")
        #print(fock_l,fock_r)
        
        configs_l = ci_vector[fock_l]
        configs_r = ci_vector[fock_r]
        
        Hblock = np.zeros((len(configs_l),len(configs_r)))

        delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
        try:
            terms = clustered_ham.terms[delta_fock]
        except KeyError:
            return 
        for term in terms:
            # Compute the state sign now - since it only depends on fock spaces
            state_sign = 1

            term_exists = True
            for oi,o in enumerate(term.ops):
                if o == '':
                    continue
                if len(o) == 1 or len(o) == 3:
                    for cj in range(oi):
                        state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
            
                # Check to make sure each cluster is allowed to make the requested transition
                try:
                    do = clusters[oi].ops[o]
                except:
                    print(" Couldn't find:", term)
                    exit()
                try:
                    d = do[(fock_l[oi],fock_r[oi])]
                    #d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
                except:
                    term_exists = False
            if not term_exists:
                continue 

            for config_li, config_l in enumerate(configs_l):
                idx_l = config_li 
                #idx_l = fock_space_shifts[fock_li] + config_li 
                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = config_ri 
                    #idx_r = fock_space_shifts[fock_ri] + config_ri 
                    
                    if diagonal and idx_r<idx_l:
                        continue
            
                    # Check to make sure each cluster is diagonal except if active
                    allowed = True
                    for ci in range(n_clusters):
                        if (config_l[ci]!=config_r[ci]) and (ci not in term.active):
                            allowed = False
                    if not allowed:
                        continue
                   
                    me = term.matrix_element(fock_l,config_l,fock_r,config_r)
#                    #d = do[(fock_bra[oi],fock_ket[oi])][bra[oi],ket[oi]] #D(I,J,:,:...)
#                    mats = []
#                    for ci in term.active:
#                        mats.append( clusters[ci].ops[term.ops[ci]][(fock_l[ci],fock_r[ci])][config_l[ci],config_r[ci]] ) 
#
#                    me = 0.0
#                  
#                    if len(mats) != len(term.active):
#                        continue
#                    
#                    #check that the mats where treated as views and also contiguous
#                    #for m in mats:
#                    #    print(m.flags['OWNDATA'])  #False -- apparently this is a view
#                    #    print(m.__array_interface__)
#                    #    print()
#
#                    # todo:
#                    #    For some reason, precompiled contract expression is slower than direct einsum - figure this out
#                    #me = term.contract_expression(*mats) * state_sign
#                    me = np.einsum(term.contract_string,*mats,term.ints) * state_sign

                    Hblock[idx_l,idx_r] += me
                   
                    if diagonal and idx_r>idx_l:
                        Hblock[idx_r,idx_l] += me
        return Hblock
        



    fock_space_shifts = [0]
    for fi,f in enumerate(ci_vector.fblocks()):
        configs_i = ci_vector[f]
        fock_space_shifts.append(fock_space_shifts[-1]+len(configs_i))


    fock_space_blocks = []
    for fock_li, fock_l in enumerate(ci_vector.data):
        for fock_ri, fock_r in enumerate(ci_vector.data):
            if fock_li > fock_ri:
                continue 
            fock_space_blocks.append( (fock_l, fock_li, fock_r, fock_ri))

    #for f in fock_space_blocks:
    #    compute_parallel_block(f)


    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)

    def test(f):
        fock_l  = f[0]
        fock_li = f[1]
        fock_r  = f[2]
        fock_ri = f[3]

        if fock_li > fock_ri:
            return 
        print(fock_l,fock_r)
        
        configs_l = ci_vector[fock_l]
        configs_r = ci_vector[fock_r]

    #pool.map(test, fock_space_blocks)
    Hblocks = pool.map(compute_parallel_block, fock_space_blocks)

    pool.close()
    pool.join()
    pool.clear()
    
    for fi,f in enumerate(fock_space_blocks):
        fock_l  = f[0] 
        fock_li = f[1] 
        fock_r  = f[2] 
        fock_ri = f[3]
        start_l = fock_space_shifts[fock_li]
        stop_l  = fock_space_shifts[fock_li+1]
        start_r = fock_space_shifts[fock_ri]
        stop_r  = fock_space_shifts[fock_ri+1]
        
        if np.all(Hblocks[fi]) != None:
            H[start_l:stop_l,start_r:stop_r] = Hblocks[fi]
        if fock_l != fock_r:
            if np.all(Hblocks[fi]) != None:
                H[start_r:stop_r,start_l:stop_l] = Hblocks[fi].T
            #try:
            #if np.all(Hblocks[fi]) != None:
            #try:
            #    H[start_r:stop_r,start_l:stop_l] = Hblocks[fi].T
            #except:
            #    pass
    
    return H
    

# }}}



def build_effective_operator(cluster_idx, clustered_ham, ci_vector,iprint=0):
    """
    Build effective operator, doing a partial trace over all clusters except cluster_idx
    
        H = sum_i o_i h_i
    """
# {{{
    clusters = ci_vector.clusters
    H = np.zeros((len(ci_vector),len(ci_vector)))
   
    new_op = ClusteredOperator(clustered_ham.clusters)
    shift_l = 0 
    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        if iprint > 0:
            print(fock_l)
       
        for config_li, config_l in enumerate(configs_l):
            idx_l = shift_l + config_li 
            
            shift_r = 0 
            for fock_ri, fock_r in enumerate(ci_vector.data):
                configs_r = ci_vector[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
                if fock_ri<fock_li:
                    shift_r += len(configs_r) 
                    continue
                try:
                    terms = clustered_ham.terms[delta_fock]
                except KeyError:
                    shift_r += len(configs_r) 
                    continue 
                
                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = shift_r + config_ri
                    if idx_r<idx_l:
                        continue
                    
                    for term in terms:
                        new_term = term.effective_cluster_operator(cluster_idx, fock_l, config_l, fock_r, config_r)
                shift_r += len(configs_r) 
        shift_l += len(configs_l)
    return new_op 
# }}}



def build_hamiltonian_diagonal(clustered_ham,ci_vector):
    """
    Build hamiltonian diagonal in basis in ci_vector
    """
# {{{
    clusters = ci_vector.clusters
    Hd = np.zeros((len(ci_vector)))
    
    shift = 0 
   
    idx = 0
    for fockspace, configs in ci_vector.items():
        for config, coeff in configs.items():
            delta_fock= tuple([(0,0) for ci in range(len(clusters))])
            terms = clustered_ham.terms[delta_fock]
            for term in terms:
                Hd[idx] += term.matrix_element(fockspace,config,fockspace,config)
            idx += 1

    return Hd

# }}}


def build_hamiltonian_diagonal_parallel1(clustered_ham_in,ci_vector, nproc=None):
    """
    Build hamiltonian diagonal in basis in ci_vector
    """
# {{{
    global clusters
    global clustered_ham
    print(" In build_hamiltonian_diagonal_parallel1. nproc=",nproc) 

    clustered_ham = clustered_ham_in
    clusters = ci_vector.clusters
    
    global delta_fock
    delta_fock= tuple([(0,0) for ci in range(len(clusters))])
  
    
    def do_parallel_work(v_curr):
        fockspace = v_curr[0]
        config = v_curr[1]
        coeff  = v_curr[2]
        
        terms = clustered_ham.terms[delta_fock]
        ## add diagonal energies
        tmp = 0
        for ci in clusters:
            tmp += ci.energies[fockspace[ci.idx]][config[ci.idx]]
        
        for term in terms:
            #tmp += term.matrix_element(fockspace,config,fockspace,config)
            tmp += term.diag_matrix_element(fockspace,config)
        return tmp

    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool
    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)

    print(" Using Pathos library for parallelization. Number of workers: ", pool.ncpus)

    out = pool.map(do_parallel_work, ci_vector)
    pool.close()
    pool.join()
    pool.clear()
    
    #out = pool.map(do_parallel_work, ci_vector, batches=100)
    #out = list(map(do_parallel_work, ci_vector))

    #Hdv = np.zeros((len(ci_vector)))
    #for o in out:
    #    Hdv[o[0]] = o[1]
    Hdv = np.array(out)
    
    return Hdv 

# }}}


def update_hamiltonian_diagonal(clustered_ham,ci_vector,Hd_vector):
    """
    Build hamiltonian diagonal in basis in ci_vector, 
    Use already computed values if stored in Hd_vector, otherwise compute, updating Hd_vector 
    with new values.
    """
# {{{
    clusters = ci_vector.clusters
    Hd = np.zeros((len(ci_vector)))
    
    shift = 0 
   
    idx = 0
    for fockspace, configs in ci_vector.items():
        for config, coeff in configs.items():
            delta_fock= tuple([(0,0) for ci in range(len(clusters))])
            try:
                Hd[idx] += Hd_vector[fockspace][config]
            except KeyError:
                try:
                    Hd_vector[fockspace][config] = 0 
                except KeyError:
                    Hd_vector.add_fockspace(fockspace)
                    Hd_vector[fockspace][config] = 0 
                terms = clustered_ham.terms[delta_fock]

                ## add diagonal energies
                tmp = 0
                for ci in clusters:
                    tmp += ci.energies[fockspace[ci.idx]][config[ci.idx]]
                
                for term in terms:
                    #Hd[idx] += term.matrix_element(fockspace,config,fockspace,config)
                    tmp += term.diag_matrix_element(fockspace,config)
                #print(" nick: %12.8f"%(Hd[idx]-tmp))
                Hd[idx] = tmp
                Hd_vector[fockspace][config] = Hd[idx] 
            idx += 1
    return Hd

# }}}

#def update_hamiltonian_diagonal_parallel1(clustered_ham,ci_vector,Hd_vector):
#    """
#    Build hamiltonian diagonal in basis in ci_vector, 
#    Use already computed values if stored in Hd_vector, otherwise compute, updating Hd_vector 
#    with new values.
#    """
## {{{
#    clusters = ci_vector.clusters
#    Hd = np.zeros((len(ci_vector)))
#    delta_fock = tuple([(0,0) for ci in range(len(clusters))])
#   
#    Hd_out = ci_vector.copy()
#    Hd_out.zero()
#
#    idx = 0
#    #get terms we already have
#    for fockspace, config, coeff in ci_vector:
#        try:
#            Hd[idx] = Hd_vector[fockspace][config]
#        except KeyError:
#            pass
#        idx += 1
#
#    for fockspace, config, coeff in ci_vector:
#        if fockspace not in Hd_outuuz
#        try:
#            a = Hd_vector[fockspace][config]
#        except KeyError:
#            try:
#                Hd_out[fockspace][config] = 0 
#            except KeyError:
#                Hd_vector.add_fockspace(fockspace)
#                Hd_vector[fockspace][config] = 0 
#            terms = clustered_ham.terms[delta_fock]
#
#            ## add diagonal energies
#            tmp = 0
#            for ci in clusters:
#                tmp += ci.energies[fockspace[ci.idx]][config[ci.idx]]
#            
#            for term in terms:
#                #Hd[idx] += term.matrix_element(fockspace,config,fockspace,config)
#                tmp += term.diag_matrix_element(fockspace,config)
#            #print(" nick: %12.8f"%(Hd[idx]-tmp))
#            Hd[idx] = tmp
#            Hd_vector[fockspace][config] = Hd[idx] 
#        idx += 1
#    
#
#    for fockspace, config, coeff in ci_vector:
#
#    out = list(map(do_parallel_work, ci_vector))
#  
#    for fock,conf,coeff in Hd_out:
#        Hd_vector[fock][conf] = coeff
#
#    for o in out:
#        Hd_out.add(o)
#    return Hd.get_vector()
#
## }}}

def precompute_cluster_basis_energies(clustered_ham):
    """
    For each cluster grab the local operator from clustered_ham, and store the expectation values
    for each cluster state
    """
    # {{{
    clusters = clustered_ham.clusters


    delta_fock= tuple([(0,0) for ci in range(len(clusters))])
    terms = clustered_ham.terms[delta_fock]
    for ci in clustered_ham.clusters:
        ci.energies = {}
        for fspace in ci.basis:
            dim = ci.basis[fspace].shape[1]
            ci.energies[fspace] = np.zeros((dim))

    config_ref = [0]*len(clusters)
    for ci in clusters:
        for fspace in ci.basis:
            fspace_curr = cp.deepcopy(list(delta_fock))
            fspace_curr[ci.idx] = fspace
            #print(fspace_curr)
            for config in range(ci.basis[fspace].shape[1]):
                config_curr = cp.deepcopy(config_ref)
                config_curr[ci.idx] = config
                e = 0
                for term in terms:
                    active = term.get_active_clusters()
                    if len(active) == 1 and active[0] == ci.idx:
                        e += term.matrix_element(fspace_curr,config_curr,fspace_curr,config_curr)
                    ci.energies[fspace][config] = e
# }}}

def precompute_cluster_basis_energies_old(clustered_ham):
    """
    For each cluster grab the local operator from clustered_ham, and store the expectation values 
    for each cluster state
    """
    # {{{
    for ci in clustered_ham.clusters:
        for fspace in ci.basis:
            dim = ci.basis[fspace].shape[1]
            ci.energies[fspace] = np.zeros((dim))
        opi = clustered_ham.extract_local_operator(ci.idx)
        for t in opi.terms:
            assert(len(t.ops)==1)
            if len(t.ops[0]) == 2:
                for fspace_del in ci.ops[t.ops[0]]:
                    assert(fspace_del[0] == fspace_del[1])
                    D = ci.ops[t.ops[0]][fspace_del]
                    
                    # e(I) += D(I,I,pq) H(pq) 
                    e = np.einsum('iipq,pq->i',D,t.ints)
                    try:
                        ci.energies[fspace_del[0]] += e
                    except KeyError:
                        ci.energies[fspace_del[0]] = e
            elif len(t.ops[0]) == 4:
                for fspace_del in ci.ops[t.ops[0]]:
                    assert(fspace_del[0] == fspace_del[1])
                    D = ci.ops[t.ops[0]][fspace_del]
                    
                    # e(I) += D(I,I,pqrs) H(pqrs) 
                    e = np.einsum('iipqrs,pqrs->i',D,t.ints)
                    try:
                        ci.energies[fspace_del[0]] += e
                    except KeyError:
                        ci.energies[fspace_del[0]] = e
# }}}

def build_1rdm(ci_vector):
    """
    Build 1rdm C_{I,J,K}<IJK|p'q|LMN> C_{L,M,N}
    """
    # {{{
    n_orb = ci_vector.n_orb
    dm_aa = np.zeros((n_orb,n_orb))
    dm_bb = np.zeros((n_orb,n_orb))
    clusters = ci_vector.clusters
   
    if 0:
        # build 1rdm in slow (easy) way
        dm_aa_slow = np.zeros((n_orb,n_orb))
        for i in range(n_orb):
            for j in range(n_orb):
                op = ClusteredOperator(clusters)
                h = np.zeros((n_orb,n_orb))
                h[i,j] = 1
                op.add_1b_terms(h)
                Nv = matvec1(op, ci_vector, term_thresh=0)
                Nv = ci_vector.dot(Nv)
                dm_aa_slow[i,j] = Nv
        print(" Here is the slow version:")
        print(dm_aa_slow)
        occs = np.linalg.eig(dm_aa_slow)[0]
        #[print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        with np.printoptions(precision=6, suppress=True):
            print(dm_aa_slow)

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
                        delta_conf = [config_l[i] == config_r[i] for i in range(len(clusters))] 
                        delta_conf[ci.idx] = True
                        diag = True
                        for i in delta_conf:
                            if i is False:
                                diag = False
                        
                        if diag == False:
                            continue
                        pq = ci.get_op_mel('Aa', fock[ci.idx], fock[ci.idx], config_l[ci.idx], config_r[ci.idx])*ci_vector[fock][config_l] * ci_vector[fock][config_r]
                        dm_aa[shifts[ci.idx]:shifts[ci.idx]+ci.n_orb, shifts[ci.idx]:shifts[ci.idx]+ci.n_orb] += pq
 
            #Bb terms
            if fock[ci.idx][1] > 0:
                # c(ijk...) <ijk...|p'q|lmk...> c(lmk...)
                # c(ijk...) <i|p'q|l> c(ljk...)
                for config_l in ci_vector.fblock(fock):
                    for config_r in ci_vector.fblock(fock):
                        # make sure all state indices are the same aside for clusters i and j
                        delta_conf = [config_l[i] == config_r[i] for i in range(len(clusters))] 
                        delta_conf[ci.idx] = True
                        diag = True
                        for i in delta_conf:
                            if i is False:
                                diag = False
                        
                        if diag == False:
                            continue
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
                

    print(ci_vector.norm())
    occs = np.linalg.eig(dm_aa + dm_bb)[0]
    print(" Eigenvalues of density matrix")
    [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
    print(np.trace(dm_aa + dm_bb))
    with np.printoptions(precision=6, suppress=True):
        print(dm_aa + dm_bb)
    return dm_aa, dm_bb 

# }}}


def build_brdm(ci_vector, ci_idx):
    """
    Build block reduced density matrix for cluster ci_idx
    """
    # {{{
    ci = ci_vector.clusters[ci_idx]
    rdms = OrderedDict()
    for fspace, configs in ci_vector.items():
        #print()
        #print("fspace:",fspace)
        #print()
        curr_dim = ci.basis[fspace[ci_idx]].shape[1]
        rdm = np.zeros((curr_dim,curr_dim))
        for configi,coeffi in configs.items():
            for cj in range(curr_dim):
                configj = list(cp.deepcopy(configi))
                configj[ci_idx] = cj
                configj = tuple(configj)
                #print(configi,configj,configi[ci_idx],configj[ci_idx])
                try:
                    #print(configi,configj,configi[ci_idx],configj[ci_idx],coeffi,configs[configj])
                    rdm[configi[ci_idx],cj] += coeffi*configs[configj]
                    #print(configi[ci_idx],cj,rdm[configi[ci_idx],cj])
                except KeyError:
                    pass
        try:
            rdms[fspace[ci_idx]] += rdm 
        except KeyError:
            rdms[fspace[ci_idx]] = rdm 

    return rdms
# }}}


def do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, max_iter_cmf=10, do_pt2=True):
    """
    Sort the cluster pairs based on how much correlation energy is recovered when combined
    """
# {{{
    dimer_energies = {}
    init_dim = 1
    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    for ci,c in enumerate(clusters):
        init_dim = init_dim * calc_nchk(c.n_orb,init_fspace[ci][0])
        init_dim = init_dim * calc_nchk(c.n_orb,init_fspace[ci][1])
    
    for i in range(len(blocks)):
        for j in range(i+1,len(blocks)):
            if len(blocks[i]) + len(blocks[j]) > max_cluster_size:
                continue
            
            new_block = []
            new_block.extend(blocks[i])
            new_block.extend(blocks[j])
            new_block = sorted(new_block)
            new_blocks = [new_block]
            new_init_fspace = [(init_fspace[i][0]+init_fspace[j][0],init_fspace[i][1]+init_fspace[j][1])]
            for k in range(len(blocks)):
                if k!=i and k!=j:
                    new_blocks.append(blocks[k])
                    new_init_fspace.append(init_fspace[k])
            new_init_fspace = tuple(new_init_fspace)
            
            new_clusters = []
            for ci,c in enumerate(new_blocks):
                new_clusters.append(Cluster(ci,c))
            
            new_ci_vector = ClusteredState(new_clusters)
            new_ci_vector.init(new_init_fspace)
         
            ## unless doing PT2, make sure new dimension is greater than 1
            if do_pt2 == False:
                dim = 1
                for ci,c in enumerate(new_clusters):
                    dim = dim * calc_nchk(c.n_orb,new_init_fspace[ci][0])
                    dim = dim * calc_nchk(c.n_orb,new_init_fspace[ci][1])
                if dim <= init_dim:
                    continue
            
            print(" Clusters:")
            [print(ci) for ci in new_clusters]
            
            new_clustered_ham = ClusteredOperator(new_clusters)
            print(" Add 1-body terms")
            new_clustered_ham.add_1b_terms(cp.deepcopy(h))
            print(" Add 2-body terms")
            new_clustered_ham.add_2b_terms(cp.deepcopy(g))
            #clustered_ham.combine_common_terms(iprint=1)
           
           
            # Get CMF reference
            print(" Let's do CMF for blocks %4i:%-4i"%(i,j))
            e_curr,converged = cmf(new_clustered_ham, new_ci_vector, cp.deepcopy(h), cp.deepcopy(g), max_iter=10, max_nroots=1)
           
            if do_pt2:
                e2, v = compute_pt2_correction(new_ci_vector, new_clustered_ham, e_curr)
                print(" PT2 Energy Total      = %12.8f" %(e_curr+e2))
               
                e_curr += e2

            print(" Pairwise-CMF(%i,%i) Energy = %12.8f" %(i,j,e_curr))
            dimer_energies[(i,j)] = e_curr
    
    import operator
    dimer_energies = OrderedDict(sorted(dimer_energies.items(), key=lambda x: x[1]))
    for d in dimer_energies:
        print(" || %10s | %12.8f" %(d,dimer_energies[d]))
  
    pairs = list(dimer_energies.keys())
    if len(pairs) == 0:
        return blocks, init_fspace

    #target_pair = next(iter(dimer_energies))
    target_pair = pairs[0]

    i = target_pair[0]
    j = target_pair[1]
    print(target_pair)
    new_block = []
    new_block.extend(blocks[i])
    new_block.extend(blocks[j])
    new_blocks = [new_block]
    new_init_fspace = [(init_fspace[i][0]+init_fspace[j][0],init_fspace[i][1]+init_fspace[j][1])]
    for k in range(len(blocks)):
        if k!=i and k!=j:
            new_blocks.append(blocks[k])
            new_init_fspace.append(init_fspace[k])
    print(" This is the new clustering")
    print(" | %12.8f" %dimer_energies[(i,j)], new_blocks)
    new_init_fspace = tuple(new_init_fspace)
    print(new_init_fspace)
    print()
    return new_blocks, new_init_fspace
# }}}

def compute_pt2_correction(ci_vector, clustered_ham, e0, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()
    if nproc==1:
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    pt_vector.prune_empty_fock_spaces()
    
    
    tmp = ci_vector.dot(pt_vector)
    var = pt_vector.norm() - tmp*tmp 
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
       
    if nproc==1:
        Hd = build_hamiltonian_diagonal(clustered_ham, pt_vector)
    else:
        Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
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
    return e2,pt_vector
# }}}

def run_hierarchical_sci(h,g,blocks,init_fspace,dimer_threshold,ecore):
    """
    compute a dimer calculation and figure out what states to retain for a larger calculation
    """
# {{{
    fclusters = []
    findex_list = {}
    for ci,c in enumerate(blocks):
        fclusters.append(Cluster(ci,c))
        #findex_list[c] = {}

    n_blocks = len(blocks)

    for ca in range(0,n_blocks):
        for cb in range(ca+1,n_blocks):
            f_idx = [ca,cb]
            s_blocks = [blocks[ca],blocks[cb]]
            s_fspace = ((init_fspace[ca]),(init_fspace[cb]))
            print("Blocks:",ca,cb)
            print(s_blocks)
            print(s_fspace)

            idx = [j for sub in s_blocks for j in sub]
            # h2
            h2 = h[:,idx] 
            h2 = h2[idx,:] 
            print(h2)
            g2 = g[:,:,:,idx] 
            g2 = g2[:,:,idx,:] 
            g2 = g2[:,idx,:,:] 
            g2 = g2[idx,:,:,:] 

            #do not want clusters to be wierdly indexed.
            print(len(s_blocks[0]))
            print(len(s_blocks[1]))
            s_blocks = [range(0,len(s_blocks[0])),range(len(s_blocks[0]),len(s_blocks[0])+len(s_blocks[1]))]

            s_clusters = []
            for ci,c in enumerate(s_blocks):
                s_clusters.append(Cluster(ci,c))

            #Cluster States initial guess
            ci_vector = ClusteredState(s_clusters)
            ci_vector.init((s_fspace))
            ci_vector.print_configs()
            print(" Clusters:")
            [print(ci) for ci in s_clusters]

            #Clustered Hamiltonian
            clustered_ham = ClusteredOperator(s_clusters)
            print(" Add 1-body terms")
            clustered_ham.add_1b_terms(h2)
            print(" Add 2-body terms")
            clustered_ham.add_2b_terms(g2)

            do_cmf = 0
            if do_cmf:
                # Get CMF reference
                cmf(clustered_ham, ci_vector, h2, g2, max_iter=10,max_nroots=50)
            else:
                # Get vaccum reference
                for ci_idx, ci in enumerate(s_clusters):
                    print()
                    print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
                    ci.form_eigbasis_from_ints(h2,g2,max_roots=50)
                    print(" Build these local operators")
                    print(" Build mats for cluster ",ci.idx)
                    ci.build_op_matrices()

            ci_vector.expand_to_full_space()
            H = build_full_hamiltonian(clustered_ham, ci_vector)
            vguess = ci_vector.get_vector()
            #e,v = scipy.sparse.linalg.eigsh(H,1,v0=vguess,which='SA')
            e,v = scipy.sparse.linalg.eigsh(H,1,which='SA')
            idx = e.argsort()
            e = e[idx]
            v = v[:,idx]
            v0 = v[:,0]
            e0 = e[0]
            print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))
            ci_vector.zero()
            ci_vector.set_vector(v0)
            ci_vector.print_configs()

            for fspace, configs in ci_vector.data.items():
                for ci_idx, ci in enumerate(s_clusters):
                    print("fspace",fspace[ci_idx])
                    print("ci basis old\n",ci.basis[fspace[ci_idx]])
                    vec = ci.basis[fspace[ci_idx]]
                    #print(configs.items())
                    idx = []
                    for configi,coeffi in configs.items():
                        #print(configi[ci_idx],coeffi)
                        if abs(coeffi) > dimer_threshold:
                            if configi[ci_idx] not in idx:
                                idx.append(configi[ci_idx])
                    print("IDX of Cluster")
                    print(ci_idx,f_idx[ci_idx])
                    print(idx)
                    try:
                        findex_list[f_idx[ci_idx],fspace[ci_idx]] = sorted(list(set(findex_list[f_idx[ci_idx],fspace[ci_idx]]) | set(idx)))
                        #ci.cs_idx[fspace[ci_idx]] =  sorted(list(set(ci.cs_idx[fspace[ci_idx]]) | set(idx)))
                    except:
                        #print(findex_list[ci_idx][fspace[ci_idx]])
                        findex_list[f_idx[ci_idx],fspace[ci_idx]] = sorted(idx)
                        #ci.cs_idx[fspace[ci_idx]] =  sorted(idx) 

                    ### TODO 
                    # first : have to save these indices in fcluster obtect and not the s_cluster. so need to change that,
                    # second: have to move the rest of the code in the block to outside pair loop. loop over fspace
                    #           look at indices kept for fspace and vec also is in fspace. and then prune it.

                    print(vec.shape)
                    print(findex_list[f_idx[ci_idx],fspace[ci_idx]])
                    vec = vec[:,findex_list[f_idx[ci_idx],fspace[ci_idx]]]
                    #vec = vec[:,idx]
                    print("ci basis new\n")
                    print(vec)
                    fclusters[f_idx[ci_idx]].basis[fspace[ci_idx]] = vec
                    #print(ci.basis[fspace[ci_idx]])
                    print("Fock indices")
                    print(findex_list)

    for ci_idx, ci in enumerate(fclusters):
        ci.build_op_matrices()
        #print(findex_list[ci_idx])
    print("     *====================================================================.")
    print("     |         Tensor Product Selected Configuration Interaction          |")
    print("     *====================================================================*")

    #Cluster States initial guess
    ci_vector = ClusteredState(fclusters)
    ci_vector.init((init_fspace))
    print(" Clusters:")
    [print(ci) for ci in fclusters]


    #Clustered Hamiltonian
    clustered_ham = ClusteredOperator(fclusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    ci_vector, pt_vector, etci, etci2,l  = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-6, thresh_ci_clip=5e-4,asci_clip=0)
    #ci_vector, pt_vector, etci, etci2  = bc_cipsi(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-10, thresh_ci_clip=5e-6,asci_clip=0.01)

    print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
    print(" TPSCI(2):       %12.8f      Dim:%6d" % (etci2+ecore,len(pt_vector)))
# }}}

def compute_rspt2_correction(ci_vector, clustered_ham, e0, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()
    if nproc==1:
        #h0v(clustered_ham,ci_vector)
        #exit()
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    #pt_vector.prune_empty_fock_spaces()
    
    
    tmp = ci_vector.dot(pt_vector)
    var = pt_vector.norm() - tmp*tmp 
    print(" Variance: %12.8f" % var,flush=True)
    
    
    print("Dim of PT space %4d"%len(pt_vector))
    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]
    print("Dim of PT space %4d"%len(pt_vector))
    
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])
    
    print(" Norm of CI vector = %12.8f" %ci_vector.norm())
    print(" Dimension of CI space: ", len(ci_vector))
    print(" Dimension of PT space: ", len(pt_vector))
    print(" Compute Denominator",flush=True)
    # compute diagonal for PT2
    start = time.time()

    #pt_vector.prune_empty_fock_spaces()
        
       
    if nproc==1:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
    else:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        #Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
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
    return e2,pt_vector
# }}}


def compute_lcc2_correction(ci_vector, clustered_ham, e0, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()
    if nproc==1:
        #h0v(clustered_ham,ci_vector)
        #exit()
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    #pt_vector.prune_empty_fock_spaces()
    
    tmp = ci_vector.dot(pt_vector)
    var = pt_vector.norm() - tmp*tmp 
    print(" Variance: %12.8f" % var,flush=True)
    
    print("Dim of PT space %4d"%len(pt_vector))
    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]
    print("Dim of PT space %4d"%len(pt_vector))
    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])
    
    print(" Norm of CI vector = %12.8f" %ci_vector.norm())
    print(" Dimension of CI space: ", len(ci_vector))
    print(" Dimension of PT space: ", len(pt_vector))
    print(" Compute Denominator",flush=True)
    # compute diagonal for PT2
    start = time.time()

    #pt_vector.prune_empty_fock_spaces()
        
       
    if nproc==1:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        Hd2 = build_hamiltonian_diagonal(clustered_ham, pt_vector)
    else:
        Hd = build_h0(clustered_ham, ci_vector, pt_vector)
        Hd2 = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
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

    print("E DPS %16.8f"%e0)
    #ci_vector.add(pt_vector)
    H = build_full_hamiltonian(clustered_ham, pt_vector)
    print(H)
    #np.fill_diagonal(H,0)
    #for i in range(0,H.shape[0]):
    #    H[i,i] = Hd[i]
    print(H)

    denom.shape = (denom.shape[0],1)
    v1 = pt_vector.get_vector()
    for j in range(0,10):
        print("v1",v1.shape)
        v2 = H @ v1
        #print("v2",v2.shape)
        #print("denom",denom.shape)
        v2 = np.multiply(denom,v2)
        #print("afrer denom",v2.shape)
        e3 = np.dot(pt_vector_v,v2)
        print(e3.shape)
        print("PT2",e2)
        print("PT3",e3[0])
        v1 = v2
    pt_vector.set_vector(v2)

    return e3[0],pt_vector
# }}}


def build_h0(clustered_ham,ci_vector,pt_vector):
    """
    Build hamiltonian diagonal in basis in ci_vector as difference of cluster energies as in RSPT
    """
# {{{
    clusters = ci_vector.clusters
    Hd = np.zeros((len(pt_vector)))
    
    shift = 0 
    idx = 0
    E0 = 0
    for fockspace, configs in ci_vector.items():
        for config, coeff in configs.items():
            delta_fock= tuple([(0,0) for ci in range(len(clusters))])
            terms = clustered_ham.terms[delta_fock]
            print(coeff,config)
            # for EN:
            for term in terms:
                E0 += term.matrix_element(fockspace,config,fockspace,config)
                #print(term.active)
            idx += 1
    print("E0%16.8f"%E0)
   
    idx = 0
    Hd = np.zeros((len(pt_vector)))
    for ci_fspace, ci_configs in ci_vector.items():
        for ci_config, ci_coeff in ci_configs.items():
            for fockspace, configs in pt_vector.items():
                print("FS",fockspace)
                active = []
                inactive = []
                for ind,fs in enumerate(fockspace):
                    if fs != ci_fspace[ind]:
                        active.append(ind)
                    else:
                        inactive.append(ind)
                            
                delta_fock= tuple([(fockspace[ci][0]-ci_fspace[ci][0], fockspace[ci][1]-ci_fspace[ci][1]) for ci in range(len(clusters))])
                print("active",active)
                print("active",delta_fock)
                #print(tuple(np.array(list(fockspace))[inactive]))

                for config, coeff in configs.items():
                    delta_fock= tuple([(0,0) for ci in range(len(clusters))])

                    diff = tuple(x-y for x,y in zip(ci_config,config))
                    print("CI",ci_config)
                    for x in inactive:
                        if diff[x] != 0 :
                            active.append(x)
                    print("PT",config)
                    print("d ",diff)
                    print("ACTIVE",active)

                    Hd[idx] = E0
                    for cidx in active:
                        fspace = fockspace[cidx]
                        conf = config[cidx]
                        print(" Cluster: %4d  Fock Space:%s config:%4d Energies %16.8f"%(cidx,fspace,conf,clusters[cidx].energies[fspace][conf]))
                        Hd[idx] += clusters[cidx].energies[fockspace[cidx]][config[cidx]]
                        Hd[idx] -= clusters[cidx].energies[ci_fspace[cidx]][ci_config[cidx]]
                        print("-Cluster: %4d  Fock Space:%s config:%4d Energies %16.8f"%(cidx,ci_fspace[cidx],ci_config[cidx],clusters[cidx].energies[ci_fspace[cidx]][ci_config[cidx]]))
                    # for EN:
                    #for term in terms:
                    #    Hd[idx] += term.matrix_element(fockspace,config,fockspace,config)
                    #    print(term.active)


                    # for RS
                    #Hd[idx] = E0 - term.active  
                    print(coeff,config)
                    idx += 1

    return Hd

    # }}}

def cepa(clustered_ham,ci_vector,pt_vector,cepa_shift):
# {{{
    ts = 0

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    #H00 = H[tb0.start:tb0.stop,tb0.start:tb0.stop]
    
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[ts]
  
    Ec = 0.0

    #cepa_shift = 'aqcc'
    #cepa_shift = 'cisd'
    #cepa_shift = 'acpf'
    #cepa_shift = 'cepa0'
    cepa_mit = 1
    cepa_mit = 100
    for cit in range(0,cepa_mit): 

        #Hdd = cp.deepcopy(H[tb0.stop::,tb0.stop::])
        Hdd = build_full_hamiltonian(clustered_ham,pt_vector,iprint=0)

        shift = 0.0
        if cepa_shift == 'acpf':
            shift = Ec * 2.0 / n_blocks
            #shift = Ec * 2.0 / n_sites
        elif cepa_shift == 'aqcc':
            shift = (1.0 - (n_blocks-3.0)*(n_blocks - 2.0)/(n_blocks * ( n_blocks-1.0) )) * Ec
        elif cepa_shift == 'cisd':
            shift = Ec
        elif cepa_shift == 'cepa0':
            shift = 0

        Hdd += -np.eye(Hdd.shape[0])*(E0 + shift)
        #Hdd += -np.eye(Hdd.shape[0])*(E0 + -0.220751700895 * 2.0 / 8.0)
        
        
        #Hd0 = H[tb0.stop::,tb0.start:tb0.stop].dot(V0[:,ts])
        H0d = build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0)
        Hd0 = H0d.T
        Hd0 = Hd0.dot(V0[:,ts])
        
        #Cd = -np.linalg.inv(Hdd-np.eye(Hdd.shape[0])*E0).dot(Hd0)
        #Cd = np.linalg.inv(Hdd).dot(-Hd0)
        
        Cd = np.linalg.solve(Hdd, -Hd0)
        
        print(" CEPA(0) Norm  : %16.12f"%np.linalg.norm(Cd))
        
        V0 = V0[:,ts]
        V0.shape = (V0.shape[0],1)
        Cd.shape = (Cd.shape[0],1)
        C = np.vstack((V0,Cd))
        H00d = np.insert(H0d,0,E0,axis=1)
        
        #E = V0[:,ts].T.dot(H[tb0.start:tb0.stop,:]).dot(C)
        E = V0[:,ts].T.dot(H00d).dot(C)
        
        cepa_last_vectors = C  
        cepa_last_values  = E
        
        print(" CEPA(0) Energy: %16.12f"%E)
        
        if abs(E-E0 - Ec) < 1e-10:
            print("Converged")
            break
        Ec = E - E0
    return Ec[0]
# }}}

def build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0):
    """
    Build hamiltonian in basis of two different clustered states
    """
# {{{
    clusters = ci_vector.clusters
    H0d = np.zeros((len(ci_vector),len(pt_vector)))
    
    shift_l = 0 
    for fock_li, fock_l in enumerate(ci_vector.data):
        configs_l = ci_vector[fock_l]
        if iprint > 0:
            print(fock_l)
       
        for config_li, config_l in enumerate(configs_l):
            idx_l = shift_l + config_li 
            
            shift_r = 0 
            for fock_ri, fock_r in enumerate(pt_vector.data):
                configs_r = pt_vector[fock_r]
                delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])

                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = shift_r + config_ri

                    try:
                        terms = clustered_ham.terms[delta_fock]
                    except KeyError:
                        shift_r += len(configs_r) 
                        continue 

                    #print(fock_l,fock_r,config_l,config_r)
                    #print(idx_l,idx_r)
                    for term in terms:
                        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        H0d[idx_l,idx_r] += me
                    

                """
                if fock_ri<fock_li:
                    shift_r += len(configs_r) 
                    continue
                try:
                    terms = clustered_ham.terms[delta_fock]
                except KeyError:
                    shift_r += len(configs_r) 
                    continue 
                
                for config_ri, config_r in enumerate(configs_r):        
                    idx_r = shift_r + config_ri
                    if idx_r<idx_l:
                        continue
                    
                    for term in terms:
                        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        H[idx_l,idx_r] += me
                        if idx_r>idx_l:
                            H[idx_r,idx_l] += me
                        #print(" %4i %4i = %12.8f"%(idx_l,idx_r,me),"  :  ",config_l,config_r, " :: ", term)
                """
                shift_r += len(configs_r) 
        shift_l += len(configs_l)
    return H0d
# }}}

def compute_cisd_correction(ci_vector, clustered_ham, nproc=1):
    # {{{
    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    E0,V0 = np.linalg.eigh(H00)
    E0 = E0[0]

    if nproc==1:
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
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
    
    ci_vector.add(pt_vector)
       
    if nproc==1:
        H = build_full_hamiltonian(clustered_ham, ci_vector)
    else:
        H = build_full_hamiltonian(clustered_ham, ci_vector)
    e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v = v[:,0]
    e0 = e[0]
    e = e[0]
    Ec = e - E0

    print(" CISD Energy Correction = %12.8f" %Ec)
    return Ec
# }}}
