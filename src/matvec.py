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

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

#@profile
def matvec_update_with_new_configs(coeff_tensor, new_configs, configs, active, thresh_search=1e-12):
   # {{{
    nactive = len(active) 
   
    _abs = abs


    config_curr = [i[0] for i in new_configs]
    count = 0
    if nactive==2:

        for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
            config_curr[active[0]] = I[0] 
            config_curr[active[1]] = I[1] 
            key = tuple(config_curr)
            if key not in configs:
                configs[key] = coeff_tensor[I[0],I[1]]
            else:
                configs[key] += coeff_tensor[I[0],I[1]]
            #count += 1
        #print(" nb2: size: %8i nonzero: %8i" %(coeff_tensor.size, count))

                    
    elif nactive==3:

        for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
            config_curr[active[0]] = I[0] 
            config_curr[active[1]] = I[1] 
            config_curr[active[2]] = I[2] 
            key = tuple(config_curr)
            if key not in configs:
                configs[key] = coeff_tensor[I[0],I[1],I[2]]
            else:
                configs[key] += coeff_tensor[I[0],I[1],I[2]]
            #count += 1
        #print(" nb3: size: %8i nonzero: %8i" %(coeff_tensor.size, count))

    elif nactive==4:

        for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
            config_curr[active[0]] = I[0] 
            config_curr[active[1]] = I[1] 
            config_curr[active[2]] = I[2] 
            config_curr[active[3]] = I[3] 
            key = tuple(config_curr)
            if key not in configs:
                configs[key] = coeff_tensor[I[0],I[1],I[2],I[3]]
            else:
                configs[key] += coeff_tensor[I[0],I[1],I[2],I[3]]
            #count += 1
        #print(" nb4: size: %8i nonzero: %8i" %(coeff_tensor.size, count))

    else:
        # local terms should trigger a fail since they are handled separately 
        print(" Wrong value in update_with_new_configs")
        exit()


    return 
# }}}
   

def matvec1(h,v,thresh_search=1e-12, opt_einsum=True, nbody_limit=4):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 

    serial version

    """
# {{{
    clusters = h.clusters
    #print(" Ensure TDMs are still contiguous:")
    #for ci in h.clusters:
    #    print(ci)
    #    for o in ci.ops:
    #        for fock in ci.ops[o]:
    #            if ci.ops[o][fock].data.contiguous == False:
    #                print(" Rearrange data for %5s :" %o, fock)
    #                ci.ops[o][fock] = np.ascontiguousarray(ci.ops[o][fock])

    sigma = ClusteredState()
    sigma = v.copy() 
    sigma.zero()
     
    #from numba import jit
    #@jit
    #@jit(nopython=True)
    print(" --------------------")
    print(" In matvec1:")
    print(" thresh_search   :   ", thresh_search)
    print(" nbody_limit     :   ", nbody_limit)
    print(" opt_einsum      :   ",opt_einsum, flush=True)
    
    #obsolete...
    def map_configs_local_to_global(coeff_tensor, new_configs, thresh_search, configs):
# {{{
        for sp_idx, spi in enumerate(itertools.product(*new_configs)):
            #print(" New config: %12.8f" %tmp[sp_idx], spi)
            if abs(coeff_tensor[sp_idx]) > thresh_search:
                if spi not in configs:
                    configs[spi] = tmp[sp_idx] 
                else:
                    configs[spi] += tmp[sp_idx] 
            #    #try:    
            #    #    configs[spi] += tmp[sp_idx] 
            #    #except:
            #    #    configs[spi] = tmp[sp_idx] 

        return 
# }}}
   

    

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
                if len(term.active) > nbody_limit:
                    continue
                
                #print()
                #print(term)
                #start1 = time.time()
                
                # do local terms separately
                if len(term.active) == 1:
                    #start2 = time.time()
                    
                    for conf_ri, conf_r in enumerate(v[fock_r]):
                        ci = term.active[0]
                            
                        coeff = v[fock_r][conf_r]
                        tmp = clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                        
                        new_configs = [[i] for i in conf_r] 
                        
                        new_configs[ci] = range(clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])].shape[0])
                        
                        for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                            if abs(tmp[sp_idx]) > thresh_search:
                                if spi not in configs_l:
                                    configs_l[spi] = tmp[sp_idx] 
                                else:
                                    configs_l[spi] += tmp[sp_idx] 
                    #stop2 = time.time()


                else:
                
                    state_sign = 1
                    for oi,o in enumerate(term.ops):
                        if o == '':
                            continue
                        if len(o) == 1 or len(o) == 3:
                            for cj in range(oi):
                                state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                        
                    for conf_ri, conf_r in enumerate(v[fock_r]):
                    
                        nonzeros = []
                        #nnz = 0
                
                        #print("  ", conf_r)
                        
                        #if abs(v[fock_r][conf_r]) < 5e-2:
                        #    continue
                        # get state sign 
                        #print('state_sign ', state_sign)
                        opii = -1
                        mats = []
                        good = True
                        #sparse_thresh = 1e-3
                        for opi,op in enumerate(term.ops):
                   
                            # this is from the JW code used to find non-zero transitions
                            #for i,ii in enumerate(oi[:,ket[ci_idx]]):
                            #    if ii*ii > thresh_transition:
                            #        nonzeros_curr.append(i)
                            #        nnz += 1
                            #nonzeros.append(nonzeros_curr)
                            
                            if op == "":
                                continue
                            opii += 1
                            #print(opi,term.active)
                            ci = clusters[opi]
                            #ci = clusters[term.active[opii]]
                            try:
                                oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
                                
                                #nonzeros_curr = []
                                #for K in range(oi.shape[0]):
                                #    if np.amax(np.abs(oi[K,:])) > thresh_search:
                                #        nonzeros_curr.append(K)
                                #oinz = oi[nonzeros_curr,:]
                                #mats.append(oinz)
                                #nonzeros.append(nonzeros_curr)
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
                        #start2 = time.time()
                        #print(term.contract_string_matvec)
                        tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                        #stop2 = time.time()
                        
                    
                        v_coeff = v[fock_r][conf_r]
                        #tmp = state_sign * tmp.ravel() * v_coeff
                        tmp = state_sign * tmp * v_coeff
                    
                        new_configs = [[i] for i in conf_r] 
                        for cacti,cact in enumerate(term.active):
                            #new_configs[cact] = nonzeros[cacti] 
                            new_configs[cact] = range(mats[cacti].shape[0])
                       
                        #print(new_configs)
                        #print(tmp.shape)
                        #print("kk")
                        #map_configs_local_to_global(tmp, new_configs, thresh_search, configs_l)
                        matvec_update_with_new_configs(tmp, new_configs, configs_l, term.active, thresh_search)
                        #map_configs_local_to_global(tmp, new_configs, thresh_search)
                        #map_configs_local_to_global(tmp, new_configs, thresh_search, configs_l)
                        
                        #for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        #    #print(" New config: %12.8f" %tmp[sp_idx], spi)
                        #    if abs(tmp[sp_idx]) > thresh_search:
                        #        if spi not in configs_l:
                        #            configs_l[spi] = tmp[sp_idx] 
                        #        else:
                        #            configs_l[spi] += tmp[sp_idx] 
                #stop1 = time.time()
                #print(" Time spent in einsum: %12.2f: total: %12.2f: NBody: %6i" %( stop2-start2,  stop1-start1, len(term.active)))
    
    print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    print(" --------------------")
    return sigma 
# }}}

def matvec1_parallel1(h_in,v,thresh_search=1e-12, nproc=None, nbody_limit=4, opt_einsum=True):
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
    

    sigma = ClusteredState()
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
                if len(term.active) > nbody_limit:
                    continue
                
                # do local terms separately
                if len(term.active) == 1:
                    
                    ci = term.active[0]
                        
                    tmp = clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                    
                    new_configs = [[i] for i in conf_r] 
                    
                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])].shape[0])
                    
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        if abs(tmp[sp_idx]) > thresh_search:
                            if spi not in configs_l:
                                configs_l[spi] = tmp[sp_idx] 
                            else:
                                configs_l[spi] += tmp[sp_idx] 


                else:
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
                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                    
                    
                    #v_coeff = v[fock_r][conf_r]
                    #tmp = state_sign * tmp.ravel() * v_coeff
                    tmp = state_sign * tmp.ravel() * coeff 
                    
                    new_configs = [[i] for i in conf_r] 
                    for cacti,cact in enumerate(term.active):
                        new_configs[cact] = range(mats[cacti].shape[0])
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
                        if abs(tmp[sp_idx]) > thresh_search:
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


# to drop:
def matvec1_parallel2(h_in,v,thresh_search=1e-12, nproc=None, opt_einsum=True, nbody_limit=4):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 

    loops over result of tensor contraction elementwise and returns result
    """
# {{{
    print(" ---------------------")
    print(" In matvec1_parallel2:")
    print(" thresh_search   :   ", thresh_search)
    print(" nbody_limit     :   ", nbody_limit)
    print(" opt_einsum      :   ", opt_einsum, flush=True)
    print(" nproc           :   ", nproc, flush=True)
    
    global h 
    global clusters
    global sigma 
    global clustered_ham
    h = h_in
    clusters = h_in.clusters
    

    sigma = ClusteredState()
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
                if len(term.active) > nbody_limit:
                    continue
                #print()
                #print(term)
                #start1 = time.time()
                
                # do local terms separately
                if len(term.active) == 1:
                    #start2 = time.time()
                    
                    ci = term.active[0]
                        
                    tmp = clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                    
                    new_configs = [[i] for i in conf_r] 
                    
                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])].shape[0])
                    
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        if abs(tmp[sp_idx]) > thresh_search:
                            if spi not in configs_l:
                                configs_l[spi] = tmp[sp_idx] 
                            else:
                                configs_l[spi] += tmp[sp_idx] 
                    #stop2 = time.time()


                else:
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
                    
                    #start2 = time.time()
                    #print()
                    #print(term)
                    #print('mats:', end='')
                    #[print(m.shape,end='') for m in mats]
                    #print('ints:', term.ints.shape)
                    #print("contract_string       :", term.contract_string)
                    #print("contract_string_matvec:", term.contract_string_matvec, flush=True)
                    
                    
                    #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                    

                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                    

                    #stop2 = time.time()
                    
                    
                    #v_coeff = v[fock_r][conf_r]
                    #tmp = state_sign * tmp.ravel() * v_coeff
                    tmp = state_sign * tmp.ravel() * coeff 
                    
                    _abs = abs

                    new_configs = [[i] for i in conf_r] 
                    for cacti,cact in enumerate(term.active):
                        new_configs[cact] = range(mats[cacti].shape[0])
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        #print(" New config: %12.8f" %tmp[sp_idx], spi)
                        if _abs(tmp[sp_idx]) > thresh_search:
                            if spi not in configs_l:
                                configs_l[spi] = tmp[sp_idx] 
                            else:
                                configs_l[spi] += tmp[sp_idx] 
                #stop1 = time.time()
                #print(" Time spent in einsum: %12.2f: total: %12.2f: NBody: %6i" %( stop2-start2,  stop1-start1, len(term.active)))
        return sigma_out
    
    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)
 
    #print(" This is the Hamiltonian we will process:")
    #for terms in clustered_ham.terms:
    #    print(terms)
    #    for term in clustered_ham.terms[terms]:
    #        print(term)
   

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

    #import pathos.profile as pr
    #pr.enable_profiling()

    out = pool.map(do_batch, conf_batches)
    #out = pool.map(do_parallel_work, v, batches=100)
    
    #pr.profile('cumulative', pipe=pool.pipe)(test_import, 'pox')
    #pr.disable_profiling()

    
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


def matvec1_parallel3(h_in,v,thresh_search=1e-12, nproc=None, opt_einsum=True, nbody_limit=4):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 
    
    use numpy to vectorize loops over result of tensor contraction (preferred) 
    """
# {{{
    print(" ---------------------")
    print(" In matvec1_parallel3:")
    print(" thresh_search   :   ", thresh_search)
    print(" nbody_limit     :   ", nbody_limit)
    print(" opt_einsum      :   ", opt_einsum, flush=True)
    print(" nproc           :   ", nproc, flush=True)
  
    if len(v) == 0:
        print(" Empty vector!")
        exit()  
    global h 
    global clusters
    global sigma 
    global clustered_ham
    h = h_in
    clusters = h_in.clusters
    

    sigma = ClusteredState()
    sigma = v.copy() 
    sigma.zero()
    
    def matvec_update_with_new_configs2(coeff_tensor, new_configs, configs, active, thresh_search=1e-12):
       # {{{
        nactive = len(active) 
       
        _abs = abs
    
    
        config_curr = [i[0] for i in new_configs]
        count = 0
        if nactive==2:
    
            for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                config_curr[active[0]] = I[0] 
                config_curr[active[1]] = I[1] 
                key = tuple(config_curr)
                if key not in configs:
                    configs[key] = coeff_tensor[I[0],I[1]]
                else:
                    configs[key] += coeff_tensor[I[0],I[1]]
                #count += 1
            #print(" nb2: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
                        
        elif nactive==3:
    
            for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                config_curr[active[0]] = I[0] 
                config_curr[active[1]] = I[1] 
                config_curr[active[2]] = I[2] 
                key = tuple(config_curr)
                if key not in configs:
                    configs[key] = coeff_tensor[I[0],I[1],I[2]]
                else:
                    configs[key] += coeff_tensor[I[0],I[1],I[2]]
                #count += 1
            #print(" nb3: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
        elif nactive==4:
    
            for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                config_curr[active[0]] = I[0] 
                config_curr[active[1]] = I[1] 
                config_curr[active[2]] = I[2] 
                config_curr[active[3]] = I[3] 
                key = tuple(config_curr)
                if key not in configs:
                    configs[key] = coeff_tensor[I[0],I[1],I[2],I[3]]
                else:
                    configs[key] += coeff_tensor[I[0],I[1],I[2],I[3]]
                #count += 1
            #print(" nb4: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
        else:
            # local terms should trigger a fail since they are handled separately 
            print(" Wrong value in update_with_new_configs")
            exit()
    
    
        return 
    # }}}

    def do_batch(batch):
        sigma_out = {} 
        #sigma_out = OrderedDict() 
        for v_curr in batch:

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
                    if len(term.active) > nbody_limit:
                        continue
                    #print()
                    #print(term)
                    #start1 = time.time()
                    
                    # do local terms separately
                    if len(term.active) == 1:
                        #start2 = time.time()
                        
                        ci = term.active[0]
                            
                        tmp = clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                        
                        new_configs = [[i] for i in conf_r] 
                        
                        new_configs[ci] = range(clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])].shape[0])
                        
                        for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                            if abs(tmp[sp_idx]) > thresh_search:
                                if spi not in configs_l:
                                    configs_l[spi] = tmp[sp_idx] 
                                else:
                                    configs_l[spi] += tmp[sp_idx] 
                        #stop2 = time.time()
            
            
                    else:
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
            
                        nonzeros = []
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
                                    
                                #nonzeros_curr = []
                                #for K in range(oi.shape[0]):
                                #    if np.amax(np.abs(oi[K,:])) > 1e-16:
                                #    #if np.amax(np.abs(oi[K,:])) > thresh_search/10:
                                #        nonzeros_curr.append(K)
                                #oinz = oi[nonzeros_curr,:]
                                #mats.append(oinz)
                                #nonzeros.append(nonzeros_curr)
                                mats.append(oi)
                                nonzeros.append(range(oi.shape[0]))
            
                            except KeyError:
                                good = False
                                break
                        if good == False:
                            continue                        
                            #break
                       
                        if len(mats) == 0:
                            continue
                        
                        #start2 = time.time()
                        #print()
                        #print(term)
                        #print('mats:', end='')
                        #[print(m.shape,end='') for m in mats]
                        #print('ints:', term.ints.shape)
                        #print("contract_string       :", term.contract_string)
                        #print("contract_string_matvec:", term.contract_string_matvec, flush=True)
                        
                        
                        #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                        
            
                        tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                        
            
                        #stop2 = time.time()
                        
                        
                        tmp = state_sign * tmp * coeff 
                        
                        new_configs = [[i] for i in conf_r] 
                        for cacti,cact in enumerate(term.active):
                            new_configs[cact] = nonzeros[cacti] 
                            #new_configs[cact] = range(mats[cacti].shape[0])
                            
                        matvec_update_with_new_configs2(tmp, new_configs, configs_l, term.active, thresh_search)
                    #stop1 = time.time()
                    #print(" Time spent in einsum: %12.2f: total: %12.2f: NBody: %6i" %( stop2-start2,  stop1-start1, len(term.active)))
        return sigma_out
    
    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)
 
    #print(" This is the Hamiltonian we will process:")
    #for terms in clustered_ham.terms:
    #    print(terms)
    #    for term in clustered_ham.terms[terms]:
    #        print(term)
   

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

    #import pathos.profile as pr
    #pr.enable_profiling()

    out = pool.map(do_batch, conf_batches)
    #out = pool.map(do_parallel_work, v, batches=100)
    
    #pr.profile('cumulative', pipe=pool.pipe)(test_import, 'pox')
    #pr.disable_profiling()

    
    pool.close()
    pool.join()
    pool.clear()
    #out = list(map(do_parallel_work, v))
    print(" This is how much memory is being used to store matvec results:    ",sys.getsizeof(out)) 
    for o in out:
        sigma.add(o)

    sigma.clip(thresh_search)
    sigma.prune_empty_fock_spaces()
    print(" This is how much memory is being used to store collected results: ",sys.getsizeof(sigma.data)) 
    print(" ---------------------")
    return sigma 
# }}}


def matvec1_parallel4(h_in,v,thresh_search=1e-12, nproc=None, opt_einsum=True, nbody_limit=4, shared_mem=3e9, batch_size=1, screen=1e-10):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 
    
    use numpy to vectorize loops over result of tensor contraction (preferred) 

    parallelized with Ray
    """
# {{{
    print(" ---------------------")
    print(" In matvec1_parallel4:")
    print(" thresh_search   :   ", thresh_search)
    print(" nbody_limit     :   ", nbody_limit)
    print(" opt_einsum      :   ", opt_einsum, flush=True)
    print(" nproc           :   ", nproc, flush=True)


    import ray
    if nproc==None:
        ray.init(object_store_memory=shared_mem)
    else:
        ray.init(num_cpus=nproc, object_store_memory=shared_mem)

    #time.sleep(10)
    if len(v) == 0:
        print(" Empty vector!")
        exit()  
    #h = h_in
    
    h_id        = ray.put(h_in)

    sigma = ClusteredState()
    sigma = v.copy() 
    sigma.zero()
    
    def matvec_update_with_new_configs2(coeff_tensor, new_configs, configs, active, thresh_search=1e-12):
       # {{{
        nactive = len(active) 
       
        _abs = abs
    
    
        config_curr = [i[0] for i in new_configs]
        count = 0
        if nactive==2:
   
            for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                try:
                    config_curr[active[0]] = new_configs[active[0]][I[0]] 
                    config_curr[active[1]] = new_configs[active[1]][I[1]] 
                except:
                    print()
                    print(" Tensor: ", coeff_tensor.shape)
                    print(" Nz Idx: ", I)
                    print(" new_co: ", new_configs)
                    print(" active: ", active,flush=True)
                    exit()
                key = tuple(config_curr)
                if key not in configs:
                    configs[key] = coeff_tensor[I[0],I[1]]
                else:
                    configs[key] += coeff_tensor[I[0],I[1]]
                #count += 1
            #print(" nb2: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
                        
        elif nactive==3:
    
            for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                config_curr[active[0]] = new_configs[active[0]][I[0]] 
                config_curr[active[1]] = new_configs[active[1]][I[1]] 
                config_curr[active[2]] = new_configs[active[2]][I[2]] 
                key = tuple(config_curr)
                if key not in configs:
                    configs[key] = coeff_tensor[I[0],I[1],I[2]]
                else:
                    configs[key] += coeff_tensor[I[0],I[1],I[2]]
                #count += 1
            #print(" nb3: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
        elif nactive==4:
    
            for I in np.argwhere(np.abs(coeff_tensor) > thresh_search):
                config_curr[active[0]] = new_configs[active[0]][I[0]] 
                config_curr[active[1]] = new_configs[active[1]][I[1]] 
                config_curr[active[2]] = new_configs[active[2]][I[2]] 
                config_curr[active[3]] = new_configs[active[3]][I[3]] 
                key = tuple(config_curr)
                if key not in configs:
                    configs[key] = coeff_tensor[I[0],I[1],I[2],I[3]]
                else:
                    configs[key] += coeff_tensor[I[0],I[1],I[2],I[3]]
                #count += 1
            #print(" nb4: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
        else:
            # local terms should trigger a fail since they are handled separately 
            print(" Wrong value in update_with_new_configs")
            exit()
   
        #print(" Size of ndarray:   ", sys.getsizeof(coeff_tensor))
        #print(" Size of dictionary:", sys.getsizeof(configs))
    
        return 
    # }}}

    def matvec_update_with_new_configs1(coeff_tensor, new_configs, configs, active, thresh_search=1e-12):
       # {{{
        nactive = len(active) 
       
        _abs = abs
   
        assert(len(active) == len(coeff_tensor.shape))
    
        config_curr = [i[0] for i in new_configs]
        count = 0
        if nactive==2:
        
            _range0 = range(coeff_tensor.shape[0])
            _range1 = range(coeff_tensor.shape[1])
            for I0 in _range0:
                for I1 in _range1:
                    try:
                        config_curr[active[0]] = new_configs[active[0]][I0] 
                        config_curr[active[1]] = new_configs[active[1]][I1] 
                    except:
                        print()
                        print(" Tensor: ", coeff_tensor.shape)
                        print(" Nz Idx: ", I0,I1)
                        print(" new_co: ", new_configs)
                        print(" active: ", active,flush=True)
                        exit()
                    key = tuple(config_curr)
                    if key not in configs:
                        configs[key] = coeff_tensor[I0,I1]
                    else:
                        configs[key] += coeff_tensor[I0,I1]
                    #count += 1
            #print(" nb2: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
                        
        elif nactive==3:
    
            _range0 = range(coeff_tensor.shape[0])
            _range1 = range(coeff_tensor.shape[1])
            _range2 = range(coeff_tensor.shape[2])
            for I0 in _range0:
                for I1 in _range1:
                    for I2 in _range2:
                        config_curr[active[0]] = new_configs[active[0]][I0] 
                        config_curr[active[1]] = new_configs[active[1]][I1] 
                        config_curr[active[2]] = new_configs[active[2]][I2] 
                        key = tuple(config_curr)
                        if key not in configs:
                            configs[key] = coeff_tensor[I0,I1,I2]
                        else:
                            configs[key] += coeff_tensor[I0,I1,I2]
    
        elif nactive==4:
    
            _range0 = range(coeff_tensor.shape[0])
            _range1 = range(coeff_tensor.shape[1])
            _range2 = range(coeff_tensor.shape[2])
            _range3 = range(coeff_tensor.shape[3])
            for I0 in _range0:
                for I1 in _range1:
                    for I2 in _range2:
                        for I3 in _range3:
                            config_curr[active[0]] = new_configs[active[0]][I0] 
                            config_curr[active[1]] = new_configs[active[1]][I1] 
                            config_curr[active[2]] = new_configs[active[2]][I2] 
                            config_curr[active[3]] = new_configs[active[3]][I3] 
                            key = tuple(config_curr)
                            if key not in configs:
                                configs[key] = coeff_tensor[I0,I1,I2,I3]
                            else:
                                configs[key] += coeff_tensor[I0,I1,I2,I3]
                            #count += 1
            #print(" nb4: size: %8i nonzero: %8i" %(coeff_tensor.size, count))
    
        else:
            # local terms should trigger a fail since they are handled separately 
            print(" Wrong value in update_with_new_configs")
            exit()
    
    
        return 
    # }}}

    @ray.remote
    def do_batch(batch,h):
        sigma_out = {} 
        #h = ray.get(h_id)
        for v_curr in batch:
            fock_r = v_curr[0]
            conf_r = v_curr[1]
            coeff  = v_curr[2]
            
            #sigma_out = ClusteredState(clusters)
            for terms in h.terms:
                fock_l= tuple([(terms[ci][0]+fock_r[ci][0], terms[ci][1]+fock_r[ci][1]) for ci in range(len(h.clusters))])
                good = True
                for c in h.clusters:
                    if min(fock_l[c.idx]) < 0 or max(fock_l[c.idx]) > c.n_orb:
                        good = False
                        break
                if good == False:
                    continue
                
                #print(fock_l, "<--", fock_r)
                
                #if fock_l not in sigma_out.data:
                if fock_l not in sigma_out:
                    sigma_out[fock_l] = {} 
                    #sigma_out[fock_l] = OrderedDict()
                
                configs_l = sigma_out[fock_l] 
                
                for term in h.terms[terms]:
                    if len(term.active) > nbody_limit:
                        continue
                    
                    # do local terms separately
                    if len(term.active) == 1:
                        
                        ci = term.active[0]
                            
                        tmp = h.clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                        
                        new_configs = [[i] for i in conf_r] 
                        
                        new_configs[ci] = range(h.clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])].shape[0])
                        
                        for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                            if abs(tmp[sp_idx]) > thresh_search:
                                if spi not in configs_l:
                                    configs_l[spi] = tmp[sp_idx] 
                                else:
                                    configs_l[spi] += tmp[sp_idx] 
            
            
                    else:
                        #print(" term: ", term)
                        state_sign = 1
                        for oi,o in enumerate(term.ops):
                            if o == '':
                                continue
                            if len(o) == 1 or len(o) == 3:
                                for cj in range(oi):
                                    state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                            
            
                        nonzeros = []
                        opii = -1
                        mats = []
                        good = True
                        for opi,op in enumerate(term.ops):
                            if op == "":
                                continue
                            opii += 1
                            ci = h.clusters[opi]
                            try:
                                oi = ci.ops[op][(fock_l[ci.idx],fock_r[ci.idx])][:,conf_r[ci.idx],:]
            
                            except KeyError:
                                good = False
                                break
                            
                            nonzeros_curr = []
                            for K in range(oi.shape[0]):
                                if np.amax(np.abs(oi[K,:])) > screen:
                                #if np.amax(np.abs(oi[K,:])) > thresh_search/10:
                                    nonzeros_curr.append(K)
                            if len(nonzeros_curr) == 0:
                                good = False
                                break
                            oinz = oi[nonzeros_curr,:]
                            mats.append(oinz)
                            nonzeros.append(nonzeros_curr)
                            #mats.append(oi)
                            #nonzeros.append(range(oi.shape[0]))

                        if good == False:
                            continue                        
                       
                        if len(mats) == 0:
                            continue
                        
                        I = term.ints * state_sign * coeff 
                        tmp = np.einsum(term.contract_string_matvec, *mats, I, optimize=opt_einsum)
                        
                        #tmp = np.einsum(term.contract_string_matvec, *mats, term.ints, optimize=opt_einsum)
                        #tmp = state_sign * tmp * coeff 
                        
                        new_configs = [[i] for i in conf_r] 
                        for cacti,cact in enumerate(term.active):
                            new_configs[cact] = nonzeros[cacti] 
                            #new_configs[cact] = range(mats[cacti].shape[0])
                            
                        matvec_update_with_new_configs2(tmp, new_configs, configs_l, term.active, thresh_search)
            #print(" Size of sigma_out:", sys.getsizeof(sigma_out))
        return sigma_out
    
    # define batches
    conf_batches = []
    batch_size = min(batch_size,len(v))  
    #batch_size = math.ceil(len(v)/(2)) 
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



    result_ids = [do_batch.remote(i,h_in) for i in conf_batches]

     
    if 0:
        out = ray.get(result_ids)
        for o in out:
            sigma.add(o)
    else:

        # Combine results as soon as they finish
        #def process_incremental(sigma, result, update,nbatches):
        def process_incremental(sigma, result):
            sigma.add(result)
            result = {} # drop that memory (hopefully)
            print(".",end='',flush=True)
            #if update > 1:
            #    print(".",end='',flush=True)
            #    update = 0
            #else:
            #    update += 1/nbatches


        print(" Number of configs: ", len(v))
        print(" Number of batches: ", len(conf_batches))
        print(" Batches complete : " )
        #print("|                                                                                                    |")
        nbatches = len(conf_batches)
        update = 0
        while len(result_ids): 
            done_id, result_ids = ray.wait(result_ids) 
            #sigma = process_incremental(sigma, ray.get(done_id[0]))
            process_incremental(sigma, ray.get(done_id[0]))
            #process_incremental(sigma, ray.get(done_id[0]),update,nbatches)
            #sigma.add(ray.get(done_id[0]))
    
        print()
    ray.shutdown()
    sigma.clip(thresh_search)
    sigma.prune_empty_fock_spaces()
    print(" ---------------------")
    return sigma 
# }}}


def heat_bath_search(h_in,v,thresh_cipsi=None, nproc=None):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 
    """
# {{{
    print(" NYI: need fix")
    exit()
    global h 
    global clusters
    global sigma 
    print(" In heat_batch_search. nproc=",nproc) 
    h = h_in
    clusters = h_in.clusters
    

    sigma = ClusteredState()
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
                
                # do local terms separately
                if len(term.active) == 1:
                    
                    ci = term.active[0]
                        
                    tmp = clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])][:,conf_r[ci]] * coeff 
                    
                    new_configs = [[i] for i in conf_r] 
                    
                    new_configs[ci] = range(clusters[ci].ops['H'][(fock_l[ci],fock_r[ci])].shape[0])
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        if abs(tmp[sp_idx]) > sqrt_thresh_cipsi:
                            if spi not in configs_l:
                                configs_l[spi] = tmp[sp_idx] 
                            else:
                                configs_l[spi] += tmp[sp_idx] 
                    

                # now do non-local terms 
                else:
                    #print(" term: ", term)
                    state_sign = 1
                    for oi,o in enumerate(term.ops):
                        if o == '':
                            continue
                        if len(o) == 1 or len(o) == 3:
                            for cj in range(oi):
                                state_sign *= (-1)**(fock_r[cj][0]+fock_r[cj][1])
                        
                    #print("  ", conf_r)
                    
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
                    
                    
                    #tmp = oe.contract(term.contract_string_matvec, *mats, term.ints)
                    tmp = np.einsum(term.contract_string_matvec, *mats, term.ints)
                    
                    tmp = state_sign * tmp.ravel() * coeff 
                    
                    new_configs = [[i] for i in conf_r] 
                    for cacti,cact in enumerate(term.active):
                        new_configs[cact] = range(mats[cacti].shape[0])
                    for sp_idx, spi in enumerate(itertools.product(*new_configs)):
                        if abs(tmp[sp_idx]) > sqrt_thresh_cipsi:
                            if spi not in configs_l:
                                configs_l[spi] = tmp[sp_idx] 
                            else:
                                configs_l[spi] += tmp[sp_idx] 

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
