import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from hubbard_fn import *
from ci_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *

import pyscf
ttt = time.time()

n_orb = 6
U = 4
beta = 1.0

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
np.random.seed(2)
tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
h += tmp + tmp.T

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)


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

blocks = [[0,1],[2,3]]
blocks = [[0,1,2],[3],[4,5]]
n_blocks = len(blocks)
clusters = []

for ci,c in enumerate(blocks):
    clusters.append(Cluster(ci,c))

ci_vector = ClusteredState(clusters)
#ci_vector.init(((2,2),(0,0)))
ci_vector.init(((3,3),(0,0),(0,0)))

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
    print(" Extract local operator for cluster",ci.idx)
    opi = clustered_ham.extract_local_operator(ci_idx)
    print()
    print()
    print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
    ci.form_eigbasis_from_local_operator(opi,max_roots=1000)

#clustered_ham.add_ops_to_clusters()
print(" Build these local operators")
for c in clusters:
    print(" Build mats for cluster ",c.idx)
    c.build_op_matrices()


sigma = matvec1(clustered_ham, ci_vector)
sigma.print_configs()
sigma_v = sigma.vector()

ci_vector.expand_to_full_space()
ci_vector.print()





print(" Build full Hamiltonian")
print(" Total dimension of CI space", len(ci_vector))
sys.stdout.flush()

H = np.zeros((len(ci_vector),len(ci_vector)))

shift_l = 0 
for fock_li, fock_l in enumerate(ci_vector.data):
    configs_l = ci_vector[fock_l]
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
   

for hi in range(H.shape[0]):
    print(" %12.8f" %(H[hi,0]-sigma_v[hi]))

print(" Diagonalize Hamiltonian Matrix:")
e,v = np.linalg.eigh(H)
idx = e.argsort()   
e = e[idx]
v = v[:,idx]
v0 = v[:,0]
print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))


