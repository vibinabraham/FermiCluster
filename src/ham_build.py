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

def build_full_hamiltonian(clustered_ham,ci_vector,iprint=0, opt_einsum=True):
    """
    Build hamiltonian in basis in ci_vector
    """
# {{{
    clusters = clustered_ham.clusters
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
    print("OBSOLETE: build_full_hamiltonian_open")
    exit()
    clusters = clustered_ham.clusters
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


def build_full_hamiltonian_parallel1(clustered_ham_in,ci_vector_in,iprint=1, nproc=None, opt_einsum=True):
    """
    Build hamiltonian in basis in ci_vector

    parallelized over fock space blocks -- inefficient
    """
# {{{
    global clusters
    global ci_vector
    global clustered_ham
    
    print(" In build_full_hamiltonian_parallel1. nproc=",nproc) 

    clustered_ham = clustered_ham_in
    ci_vector = ci_vector_in
    clusters = clustered_ham_in.clusters

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



def build_full_hamiltonian_parallel2(clustered_ham_in,ci_vector_in,iprint=1, nproc=None, opt_einsum=True, thresh=1e-14):
    """
    Build hamiltonian in basis in ci_vector

    parallelized over matrix elements
    """
# {{{
    global clusters
    global ci_vector
    global clustered_ham
    
    print(" In build_full_hamiltonian_parallel2. nproc=",nproc) 

    clustered_ham = clustered_ham_in
    clusters = clustered_ham_in.clusters
    ci_vector = ci_vector_in

    H = np.zeros((len(ci_vector),len(ci_vector)))
    n_clusters = len(clusters)


    def do_parallel_work(v_curr):
        fock_l = v_curr[0]
        conf_l = v_curr[1]
        idx_l  = v_curr[2]

        out = []
        
        idx_r = -1 
        for fock_r in ci_vector.fblocks():
            confs_r = ci_vector[fock_r]
            delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
            try:
                terms = clustered_ham.terms[delta_fock]

            except KeyError:
                idx_r += len(confs_r) 
                continue
               

            for conf_r in confs_r:        
                idx_r += 1
                
                if idx_l > idx_r:
                    continue
               
                me = 0
                for term in terms:
                    me += term.matrix_element(fock_l,conf_l,fock_r,conf_r)

                #if abs(me) > thresh:
                out.append( (idx_r, me) )


        return out 
#    def parallel_work(inp):
#        fock_l = inp[0]
#        fock_r = inp[1]
#        conf_l = inp[2]
#        conf_r = inp[3]
#        idx_l  = inp[4]
#        idx_r  = inp[5]
#        out = [idx_l, idx_r, None]
#
#        delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
#        try:
#            terms = clustered_ham.terms[delta_fock]
#                
#                for config_ri, config_r in enumerate(configs_r):        
#                    idx_r = shift_r + config_ri
#                    if idx_r<idx_l:
#                        continue
#                    
#                    for term in terms:
#                        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
#                        H[idx_l,idx_r] += me
#                        if idx_r>idx_l:
#                            H[idx_r,idx_l] += me
#                        #print(" %4i %4i = %12.8f"%(idx_l,idx_r,me),"  :  ",config_l,config_r, " :: ", term)
#        
#        except KeyError:
#            continue 

    

    rows = []
    idx_row = 0
    for fock1,conf1,coeff1 in ci_vector:
        rows.append( (fock1, conf1, idx_row))
        idx_row += 1



    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool

    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)


    Hrows = pool.map(do_parallel_work, rows)

    pool.close()
    pool.join()
    pool.clear()


    for row_idx, row in enumerate(Hrows):
        for col_idx, term in row:
            assert( col_idx >= row_idx)
            H[row_idx, col_idx] = term
            H[col_idx, row_idx] = term

    
    return H
    

# }}}



def grow_hamiltonian_parallel_ray(h_old,clustered_ham,ci_vector,ci_vector_old,iprint=1, nproc=None, opt_einsum=True, thresh=1e-14,
        shared_mem=1e9):
    """
    Grow the Hamiltonian matrix by building only the new matrix elements for the new space indicated by ci_vector
    parallelized over matrix elements
    """
# {{{
    import ray
    if nproc==None:
        ray.init(object_store_memory=shared_mem)
    else:
        ray.init(num_cpus=nproc, object_store_memory=shared_mem)
    old_dim = len(ci_vector_old) 
    old_basis = ci_vector_old.copy()
    new_basis = ci_vector.copy()
    full_basis = ci_vector.copy()
    old_basis.set_vector(np.array(range(len(old_basis))))
    new_basis.set_vector(np.array(range(len(new_basis))))
    full_basis.set_vector(np.array(range(len(full_basis))))
    for f,c,v in old_basis:
        del new_basis[f][c]
    new_basis.prune_empty_fock_spaces()
    print(" Size of old space:", len(old_basis))
    print(" Size of new space:", len(new_basis))
    print(" Size of all space:", len(full_basis))
    assert(len(full_basis)==len(old_basis)+len(new_basis))
    
    clusters = clustered_ham.clusters
    print(" In grow_hamiltonian_parallel. nproc=",nproc) 

    H = np.zeros((len(ci_vector),len(ci_vector)))
    n_clusters = len(clusters)

    #for f1,c1,i1 in old_basis:
    #    for f2,c2,i2 in old_basis:
    #        H[full_basis[f1][c1],full_basis[f2][c2]] = h_old[i1,i2]
    #        print(full_basis[f1][c1],full_basis[f2][c2] , i1,i2)
    for f1,cs1 in old_basis.items():
        for c1,i1 in old_basis[f1].items():
            for f2,cs2 in old_basis.items():
                for c2,i2 in old_basis[f2].items():
                    H[full_basis[f1][c1],full_basis[f2][c2]] = h_old[i1,i2]

    for f1,c1,i1 in new_basis:
        assert(new_basis[f1][c1] == full_basis[f1][c1])
    for f1,c1,i1 in old_basis:
        old_basis[f1][c1] = full_basis[f1][c1]
        if f1 in new_basis:
            assert(c1 not in new_basis[f1])
  
    h_id            = ray.put(clustered_ham)
    new_basis_id    = ray.put(new_basis)

    try:
        assert(np.amax(np.abs(H-H.T))<1e-14)
    except AssertionError:
        for f1,c1,i1 in full_basis:
            for f2,c2,i2 in full_basis:
                if abs(H[i1,i2] - H[i2,i1])>1e-14:
                    print(f1,c1,i1)
                    print(f2,c2,i2)
                    print(H[i1,i2] - H[i2,i1])
        raise AssertionError
  

    @ray.remote
    def do_parallel_work(fock_l, conf_l, idx_l, basis_r, _h):
        out = []
        for fock_r in basis_r.fblocks():
            confs_r = basis_r[fock_r]
            delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(_h.clusters))])
            if delta_fock in _h.terms:
                for conf_r in confs_r:        
                    idx_r =  basis_r[fock_r][conf_r]
                    if idx_l <= idx_r:
                        me = 0
                        for term in _h.terms[delta_fock]:
                            me += term.matrix_element(fock_l,conf_l,fock_r,conf_r)
                        out.append( (idx_r, me) )

        return (idx_l,out)

    rows = []
    idx_row = 0
    for fock1,conf1,coeff1 in ci_vector:
        rows.append( (fock1, conf1, idx_row))
        idx_row += 1



    #import multiprocessing as mp
    #from pathos.multiprocessing import ProcessingPool as Pool
    #if nproc == None:
    #    pool = Pool()
    #else:
    #    pool = Pool(processes=nproc)


    #Hrows = pool.map(do_parallel_work, rows)
    result_ids = [do_parallel_work.remote(i[0],i[1],i[2],new_basis,h_id) for i in new_basis]
    result_ids.extend( [do_parallel_work.remote(i[0],i[1],i[2],new_basis,h_id) for i in old_basis])
    result_ids.extend( [do_parallel_work.remote(i[0],i[1],i[2],old_basis,h_id) for i in new_basis])
    
    if 1:
        for result in ray.get(result_ids):
            (row_idx,row) = result
            for col_idx, term in row:
                assert( col_idx >= row_idx)
                assert( abs(H[row_idx,col_idx])<1e-16)
                assert( abs(H[col_idx,row_idx])<1e-16)
                H[row_idx, col_idx] = term
                H[col_idx, row_idx] = term

    if 0:
        print(" Number of batches: ", len(rows))
        print(" Batches complete : " )
        # Combine results as soon as they finish
        def process_incremental(H, result):
            (row_idx,row) = result
            for col_idx, term in row:
                assert( col_idx >= row_idx)
                H[row_idx, col_idx] = term
                H[col_idx, row_idx] = term
            print(".",end='',flush=True)
        
        while len(result_ids): 
            done_id, result_ids = ray.wait(result_ids) 
            process_incremental(H, ray.get(done_id[0]))
    try:
        assert(np.amax(np.abs(H-H.T))<1e-14)
    except AssertionError:
        for f1,c1,i1 in full_basis:
            for f2,c2,i2 in full_basis:
                if abs(H[i1,i2] - H[i2,i1])>1e-14:
                    print(f1,c1,i1)
                    print(f2,c2,i2)
                    print(H[i1,i2] - H[i2,i1])
        raise AssertionError
   
    ray.shutdown()
    return H
    

# }}}



def grow_hamiltonian_parallel(h_old,clustered_ham,ci_vector,ci_vector_old,iprint=1, nproc=None, opt_einsum=True, thresh=1e-14):
    """
    Grow the Hamiltonian matrix by building only the new matrix elements for the new space indicated by ci_vector
    parallelized over matrix elements
    """
# {{{
    global _h
    old_dim = len(ci_vector_old) 
    old_basis = ci_vector_old.copy()
    new_basis = ci_vector.copy()
    full_basis = ci_vector.copy()
    old_basis.set_vector(np.array(range(len(old_basis))))
    new_basis.set_vector(np.array(range(len(new_basis))))
    full_basis.set_vector(np.array(range(len(full_basis))))
    for f,c,v in old_basis:
        del new_basis[f][c]
    new_basis.prune_empty_fock_spaces()
    print(" Size of old space:", len(old_basis))
    print(" Size of new space:", len(new_basis))
    print(" Size of all space:", len(full_basis))
    assert(len(full_basis)==len(old_basis)+len(new_basis))
    
    clusters = clustered_ham.clusters
    print(" In grow_hamiltonian_parallel. nproc=",nproc) 

    H = np.zeros((len(ci_vector),len(ci_vector)))
    n_clusters = len(clusters)

    #for f1,c1,i1 in old_basis:
    #    for f2,c2,i2 in old_basis:
    #        H[full_basis[f1][c1],full_basis[f2][c2]] = h_old[i1,i2]
    #        print(full_basis[f1][c1],full_basis[f2][c2] , i1,i2)
    for f1,cs1 in old_basis.items():
        for c1,i1 in old_basis[f1].items():
            for f2,cs2 in old_basis.items():
                for c2,i2 in old_basis[f2].items():
                    H[full_basis[f1][c1],full_basis[f2][c2]] = h_old[i1,i2]

    for f1,c1,i1 in new_basis:
        assert(new_basis[f1][c1] == full_basis[f1][c1])
    for f1,c1,i1 in old_basis:
        old_basis[f1][c1] = full_basis[f1][c1]
        if f1 in new_basis:
            assert(c1 not in new_basis[f1])
 
    _h  = clustered_ham
    new_basis_id    = ray.put(new_basis)

    try:
        assert(np.amax(np.abs(H-H.T))<1e-14)
    except AssertionError:
        for f1,c1,i1 in full_basis:
            for f2,c2,i2 in full_basis:
                if abs(H[i1,i2] - H[i2,i1])>1e-14:
                    print(f1,c1,i1)
                    print(f2,c2,i2)
                    print(H[i1,i2] - H[i2,i1])
        raise AssertionError
  

    def do_parallel_work(fock_l, conf_l, idx_l, basis_r):
        out = []
        for fock_r in basis_r.fblocks():
            confs_r = basis_r[fock_r]
            delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(_h.clusters))])
            if delta_fock in _h.terms:
                for conf_r in confs_r:        
                    idx_r =  basis_r[fock_r][conf_r]
                    if idx_l <= idx_r:
                        me = 0
                        for term in _h.terms[delta_fock]:
                            me += term.matrix_element(fock_l,conf_l,fock_r,conf_r)
                        out.append( (idx_r, me) )

        return (idx_l,out)


    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)

    jobs = [(i[0],i[1],i[2],new_basis) for i in new_basis]
    jobs.extend( [(i[0],i[1],i[2],new_basis) for i in old_basis])
    jobs.extend( [(i[0],i[1],i[2],old_basis) for i in new_basis])
    
    results = pool.map(do_parallel_work, jobs)
    
    pool.close()
    pool.join()
    pool.clear()

    for row_idx, row in enumerate(results):
        for col_idx, term in row:
            assert( col_idx >= row_idx)
            assert( abs(H[row_idx,col_idx])<1e-16)
            assert( abs(H[col_idx,row_idx])<1e-16)
            H[row_idx, col_idx] = term
            H[col_idx, row_idx] = term
    
    return H
    

# }}}



def build_effective_operator(cluster_idx, clustered_ham, ci_vector,iprint=0):
    """
    Build effective operator, doing a partial trace over all clusters except cluster_idx
    
        H = sum_i o_i h_i
    """
# {{{
    clusters = clustered_ham.clusters
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
    clusters = clustered_ham.clusters
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


def build_hamiltonian_diagonal_parallel1(clustered_ham_in, ci_vector, nproc=None):
    """
    Build hamiltonian diagonal in basis in ci_vector
    """
# {{{
    global clusters
    global clustered_ham
    print(" In build_hamiltonian_diagonal_parallel1. nproc=",nproc) 

    clustered_ham = clustered_ham_in
    clusters = clustered_ham_in.clusters
    
    global delta_fock
    delta_fock= tuple([(0,0) for ci in range(len(clusters))])
  
    
    def do_parallel_work(v_curr):
        fockspace = v_curr[0]
        config = v_curr[1]
        coeff  = v_curr[2]
        
        terms = clustered_ham.terms[delta_fock]
        ## add diagonal energies
        tmp = 0
        
        for term in terms:
            #tmp += term.matrix_element(fockspace,config,fockspace,config)
            tmp += term.diag_matrix_element(fockspace,config,opt_einsum=False)
        return tmp

    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool
    
    if nproc == None:
        pool = Pool()
    else:
        pool = Pool(processes=nproc)

    print(" Using Pathos library for parallelization. Number of workers: ", pool.ncpus)

    #chunksize = 100
    #print(" Chunksize: ", chunksize)
    #out = pool.map(do_parallel_work, ci_vector, chunksize=chunksize)
    if len(ci_vector) == 0:
        return np.array([])
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
    clusters = clustered_ham.clusters
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
    print("OBSOLETE: precompute_cluster_basis_energies")
    exit()
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
