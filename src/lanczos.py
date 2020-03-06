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

def lanczos_old(A,vi,max_iter=20):
    # {{{
    wi = A@vi 
    ai = (vi.T @ wi)
    wi = wi - vi*ai 
    bj = np.linalg.norm(wi)
    print(" Residual Norm: %12.8f " %bj)
   
    T = np.array([[ai]])

    print(T.shape)
    vj = wi/bj 

    vecs = [vi]
    
    for vv in vecs:
        s = vv.dot(vj)
        vj -=  s* vv 
    vj /=  np.linalg.norm(vj) 
    vecs.append(vj)
   
   
    for j in range(max_iter):
        wj = A@vj 
        aj = (wj.T @ vj)

        
        if j == 0:
            wj = wj - aj*vj
        else:
            wj = wj - aj*vj - bj*vi

        for l in range(j):
            wj = wj - a[l]*vecs[l] 
        bk = np.linalg.norm(wj)
        vk = wj / bk 
        
        for vv in vecs:
            s = vv.dot(vk)
            vk -= s* vv
        vk = vk / np.linalg.norm(vk) 
        vecs.append(vk)
       
        T = np.hstack((T,np.zeros([T.shape[0],1])))
        T = np.vstack((T,np.zeros([1,T.shape[1]])))
        T[j,j] = aj
        T[j,j-1] = bj
        T[j-1,j] = bj

        V = np.array(vecs).T
        T = V.T @ A @ V
        #print("T: ")
        #np.set_printoptions(suppress=True)
        #print(T)
        #print("T: ")
        e,x = np.linalg.eig(T)
        idx = e.argsort()
        e = e[idx]
        x = x[:,idx]
        print(" Current electronic energy: %12.8f  ||Res|| %12.8f" %(e[0],bk))
        
        bj = bk
        vj = vk
        vi = vj
        bi = bj
# }}}

def lanczos(A,x,max_iter=52, thresh=1e-8):
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

np.random.seed(2)
A = np.random.random((1000,1000)) - .5
A = A+A.T
v = np.random.random((1000))
v = v /np.linalg.norm(v) 
lanczos(A,v)
e,v = np.linalg.eig(A)
idx = e.argsort()
e = e[idx]
v = v[:,idx]
e = e[0:10]
print(" exact soln")
for ei in e:
    print(" %12.8f " %ei)
exit()

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

prune1 = 1e-16

vi = ci_vector 
wi = matvec1_parallel2(clustered_ham, vi)
wi.prune_empty_fock_spaces()
ai = vi.dot2(wi)
tmp = vi.copy()
tmp.scale(-1*ai)
wi.add(tmp)
bj = wi.norm()
print(" Residual Norm: %12.8f Residual Dimension: %-8i" % (bj,len(wi)))
wi.clip(prune1)
print(" Residual Norm: %12.8f Residual Dimension: %-8i" % (wi.norm(),len(wi)))
vj = wi
vj.normalize()

vecs = [vi]
max_iter = 6
T = np.array([[ai]])
for j in range(1,max_iter):
    wj = matvec1_parallel2(clustered_ham, vj)
    wj.prune_empty_fock_spaces()
    aj = vj.dot2(wj)
    
    T = np.hstack((T,np.zeros([T.shape[0],1])))
    T = np.vstack((T,np.zeros([1,T.shape[1]])))
    T[j,j] = aj
    T[j,j-1] = bj
    T[j-1,j] = bj
    print(T)

    e,x = np.linalg.eig(T)
    idx = e.argsort()
    e = e[idx]
    x = x[:,idx]

    print(" Current electronic energy: %12.8f " %(e[0]))
    print(" Current total energy:      %12.8f " %(e[0]+ecore))
    
    tmp = vj.copy()
    tmp.scale(-1*aj)
    wj.add(tmp)
    bk = wj.norm()
    print(" Residual Norm: %12.8f Residual Dimension: %-8i" % (bk,len(wj)))
    wj.clip(prune1)
    print(" Residual Norm: %12.8f Residual Dimension: %-8i" % (wj.norm(),len(wj)))
    vk = wj
    vk.normalize()
    for vv in vecs:
        ovlp = vk.dot2(vv)
        tmp = vv.copy()
        tmp.scale(-1*ovlp)
        vk.add(tmp)
    vk.normalize()
    vecs.append(vk)
    
    print(" Check orthog")
    for vv in vecs:
        print(" Overlap: %12.8f" %vk.dot2(vv))
    vk.normalize()
    vj = vk
    bj = bk


print(" FCI     total energy:      %12.8f " %(efci))
    
