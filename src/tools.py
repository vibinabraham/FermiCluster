import numpy as np
import scipy
import itertools
import copy as cp
from helpers import *
import opt_einsum as oe
import tools

from ClusteredOperator import *
from ClusteredState import *


def cmf(clustered_ham, ci_vector, h, g, max_iter=20, thresh=1e-8, max_nroots=1000):
    """ Do CMF for a tensor product state 
       
        This modifies the data in clustered_ham.clusters, both the basis, and the operators

        h is the 1e integrals
        g is the 2e integrals

        thresh: how tightly to converge energy
        max_nroots: what is the max number of cluster states to allow in each fock space of each cluster
                    after the potential is optimized?
    """
  # {{{
    rdm_a = None
    rdm_b = None
    converged = False
    clusters = clustered_ham.clusters
    e_last = 999
    for cmf_iter in range(max_iter):
        
        print(" Build cluster basis and operators")
        for ci_idx, ci in enumerate(clusters):
            assert(ci_idx == ci.idx)
            if cmf_iter > 0:
                ci.form_eigbasis_from_ints(h,g,max_roots=max_nroots, rdm1_a=rdm_a, rdm1_b=rdm_b)
            else: 
                ci.form_eigbasis_from_ints(h,g,max_roots=max_nroots)
        
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


    for fock_ri, fock_r in enumerate(v.fblocks()):

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
        
        sigma_out = ClusteredState(clusters)
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
            
            if fock_l not in sigma_out.data:
                sigma_out.add_fockspace(fock_l)
        
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
        return sigma_out.data
    
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
   
    for o in out:
        sigma.add(o)

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
  
    global glob_test
    glob_test = 'nick'
    global Hd
    Hd = ci_vector.copy()
    Hd.zero()
   
    idx = 0
    for fock,conf,coeff in ci_vector:
        Hd[fock][conf] = idx
        idx += 1
    
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
            #Hd_out[fockspace][config] = term.diag_matrix_element(fockspace,config)
            tmp += term.diag_matrix_element(fockspace,config)
        return tmp

    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)

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
   
    if 1:
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
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
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

