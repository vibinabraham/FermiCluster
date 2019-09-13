import numpy as np
import scipy
import itertools as it
import time
from math import factorial
import copy as cp

from hubbard_fn import *
from ci_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *

ttt = time.time()

n_orb = 8
U = 1.
beta = 1.

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

blocks = [[0,1,2,3],[4,5,6,7]]
blocks = [[0,1],[2,3,4,5],[6,7]]
n_blocks = len(blocks)
clusters = []

for ci,c in enumerate(blocks):
    clusters.append(Cluster(ci,c))

print(" Clusters:")
[print(ci) for ci in clusters]

clustered_ham = ClusteredOperator(clusters)
print(" Add 1-body terms")
clustered_ham.add_1b_terms(h)


print(" Build cluster basis")
for ci_idx, ci in enumerate(clusters):
    assert(ci_idx == ci.idx)
    print(" Extract local operator for cluster",ci.idx)
    opi = clustered_ham.extract_local_operator(ci_idx)
    print()
    print()
    print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
    ci.form_eigbasis_from_local_operator(opi)

clustered_ham.add_ops_to_clusters()
print(" Build these local operators")
for c in clusters:
    print(" Build mats for cluster ",c.idx)
    c.build_op_matrices()

ci_vector = ClusteredState(clusters)
ci_vector.init(((2,2),(2,2),(0,0)))
ci_vector.print()
#[print(c.ops.keys()) for c in clusters]
#exit()
for fock_li, fock_l in enumerate(ci_vector.data):
    configs_l = ci_vector[fock_l]
    for fock_ri, fock_r in enumerate(ci_vector.data):
        configs_r = ci_vector[fock_r]
        print(" Get terms for fock block: ", fock_l,fock_r)
        delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
       
        terms = clustered_ham.terms[delta_fock]
        [print(t) for t in terms]
        if fock_li==fock_ri:
            for config_li, config_l in enumerate(configs_l):
                for config_ri, config_r in enumerate(configs_r):        
                    me = 0
                    for term in terms:
                        me_term = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        me += me_term
                        print(me_term)
                    print(me)
        else:
            delta = cp.deepcopy(delta_fock)
            for ci in range(len(clusters)):
                delta[ci][2] = 1 
            for config_li, config_l in enumerate(configs_l):
                for config_ri, config_r in enumerate(configs_r):
                    print(config_l,config_r)
        #clustered_ham.matrix_element(fock_l,fock_r,)


