import sys, os
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
from numpy.linalg import norm


def test_lanczos(A,x,max_iter=52, thresh=1e-8):
    # {{{
    dim = A.shape[0]
    q = x/norm(x)
    
    Q = np.zeros((dim,0))
    q.shape = (dim,1)
    Q = np.hstack((Q,q))
    q.shape = (dim)
    
    AQ = np.zeros((dim,0))
    
    r = A@q
    
    # add to AQ
    r.shape = (dim,1)
    AQ = np.hstack((AQ,r))
    r.shape = (dim)

    ai = r.dot(q)
    r = r -  ai*q
    bi = norm(r)
    print(" Norm of residual: %12.8f" %bi)

    for j in range(1,max_iter):
        v = 1*q
        q = r/bi
        q.shape = (dim,1)
        Q = np.hstack((Q,q))
        q.shape = (dim)

        sig = A@q
        sig.shape = (dim,1)
        AQ = np.hstack((AQ,sig))
        sig.shape = (dim)
        
        r = sig-bi*v
        aj = r.dot(q)

        r = r - aj*q
        bj = norm(r)
        #print(" Norm of residual: %12.8f" %bj)

        if bj < thresh:
            return Q
        else:
            bi = 1*bj
            ai = 1*aj
        


        T = Q.T @ A @ Q
        #print("T: ")
        #np.set_printoptions(suppress=True)
        #print(T)
        e,w = np.linalg.eig(T)
        idx = e.argsort()
        e = e[idx]
        w = w[:,idx]

        vgs = Q@w[:,0]
        res =  AQ@w[:,0] - e[0] * Q@w[:,0] 
        print(" Current electronic energy: %12.8f  ||Res|| %12.8f" %(e[0],norm(res)))

# }}}


if 0:
    np.random.seed(2)
    A = np.random.random((100,100)) - .5
    A = A+A.T
    v = np.random.random((100))
    v = v /np.linalg.norm(v) 
    test_lanczos(A,v)
    e,v = np.linalg.eig(A)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    e = e[0:10]
    print(" exact soln")
    for ei in e:
        print(" %12.8f " %ei)





def sparse_lanczos(H,x, max_iter=10, thresh=1e-8, vector_prune=1e-16, sigma_prune=1e-16):
    q = x.copy()
    q.normalize()
    Q = [q.copy()] # list of subspace vectors
   
    #q.clip(prune)
    #q.normalize()
    sig = matvec1_parallel2(clustered_ham, q)
    sig.clip(sigma_prune)
    sig.prune_empty_fock_spaces()
    
    AQ = [sig] # list of sigma vectors

    r = sig.copy()
    
    ai = q.dot2(r)
    r.add(q, scalar=(-1*ai))
    bi = r.norm()

    print(" Norm of residual: %12.8f" %bi)
    
    for j in range(1,max_iter):
        v = q.copy()
        q = r.copy()
        q.normalize()
        
        q.clip(vector_prune)
        q.normalize()
        
        Q.append(q)

        for qi in Q:
            print(" Length of vector: ", len(qi))

        sig = matvec1_parallel2(clustered_ham, q)
        sig.clip(sigma_prune)
        sig.prune_empty_fock_spaces()

        AQ.append(sig)

        r = sig.copy()
        r.add(v, scalar=(-1*bi) )

        aj = q.dot2(r)

        r.add(q, scalar=(-1*aj) )

        bj = r.norm()
        

        Tdim = len(Q)
        assert(len(Q) == len(AQ))

        T = np.zeros((Tdim,Tdim))
        for ii in range(Tdim):
            for jj in range(ii,Tdim):
                T[ii,jj] = Q[ii].dot2(AQ[jj])
                T[jj,ii] = T[ii,jj]
        
        S = np.zeros((Tdim,Tdim))
        for ii in range(Tdim):
            for jj in range(ii,Tdim):
                S[ii,jj] = Q[ii].dot2(Q[jj])
                S[jj,ii] = S[ii,jj]

        #print(S)
        se,sv = np.linalg.eig(S)
        print(" Smallest eigenvalue of S: %12.8f" %min(se))
        X = np.linalg.inv(scipy.linalg.sqrtm(S))
        Torth = X.T @ T @ X
        e,w = np.linalg.eig(Torth)
        idx = e.argsort()
        e = e[idx]
        w = w[:,idx]

   
        res = x.copy()
        res.zero()
        res.clip(1)
        res.prune_empty_fock_spaces()

        for ii in range(Tdim):
            res.add(AQ[ii], scalar=w[ii,0])
            res.add(Q[ii], scalar=(-1 * e[0] * w[ii,0]))
        print(" Current electronic energy: %12.8f  ||Res|| %12.8f" %(e[0],res.norm()))
        
        if bj < thresh:
            return Q
        else:
            bi = 1*bj
            ai = 1*aj




if __name__ == "__main__":

    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000
    
    from pyscf import gto, scf, mcscf, ao2mo
    
    r0 = 1.5
    
    molecule= '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)
    
    charge = 0
    spin  = 0
    basis_set = '6-31g'
    
    ###     TPSCI BASIS INPUT
    orb_basis = 'boys'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10
    
    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    #init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1,2,3],[4,5],[6,7]]
    init_fspace = ((3, 3), (1, 1), (1, 1))
    
    
    
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)
    
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    
    #cluster using hcore
    idx = e1_order(h,cut_off = 1e-4)
    h,g = reorder_integrals(idx,h,g)
    
    
    do_fci = 1
    do_hci = 1
    do_tci = 1
    
    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    
    
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.combine_common_terms(iprint=1)
    
    
    do_cmf = 1
    if do_cmf:
        # Get CMF reference
        cmf(clustered_ham, ci_vector, h, g, max_iter=1)
        #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=50,dm_guess=(dm_aa,dm_bb),diis=True)
    
    
    
    
    sparse_lanczos(clustered_ham, ci_vector, vector_prune=1e-2, sigma_prune=1e-8)
    
    
    
    
    print(" FCI     total energy:           %12.8f " %(efci))
    print(" FCI     electronic energy:      %12.8f " %(efci-ecore))
        
    
