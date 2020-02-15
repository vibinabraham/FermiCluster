import numpy as np
import scipy
import itertools
import copy as cp
from helpers import *
import opt_einsum as oe

from ClusteredOperator import *
from ClusteredState import *

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


def matvec_open(h_in,v_in,term_thresh=1e-12):
    """
    Compute the action of H onto a sparse trial vector v
    returns a ClusteredState object. 

    """
# {{{
    global clusters
    global h 
    global v 
  
    h = h_in
    v = v_in
    clusters = h.clusters
    

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
    



    list(map(do_parallel_work, v))

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
        
        print("Processing the block: ")
        print(fock_l,fock_r)
        
        configs_l = ci_vector[fock_l]
        configs_r = ci_vector[fock_r]
        
        Hblock = np.zeros((len(configs_l),len(configs_r)))

        print(Hblock.shape)
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
            print(fspace_curr)
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
    dm_aa = np.zeros((ci_vector.n_orb,ci_vector.n_orb))
    dm_bb = np.zeros((ci_vector.n_orb,ci_vector.n_orb))


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

