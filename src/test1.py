import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp

from hubbard_fn import *
from ci_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *

ttt = time.time()

n_orb = 6
U = 1.
beta = 1.

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
tmp = np.random.rand(h.shape[0],h.shape[1])*0.001
h += tmp + tmp.T

if 0:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

blocks = [[0,1],[2,3],[4,5]]
#blocks = [[0,1,2,3],[4,5,6,7]]
#blocks = [[0,1],[2,3,4,5],[6],[7]]
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
#ci_vector.init(((2,2),(2,2),(0,0)))
#ci_vector.init(((2,2),(2,2),(0,0),(0,0)))
ci_vector.init(((1,1),(1,1),(1,1)))
#ci_vector.init(((1,1),(1,1),(1,1),(1,1)))

# add single particle transfers
fblocks = list(ci_vector.keys())
for ref_fblock in fblocks: 
    for ci in clusters:
        for cj in clusters:
            if ci.idx != cj.idx:
                # alpha transfer
                new_fblock = [[b[0],b[1]] for b in ref_fblock]
                new_fblock[ci.idx][0] += 1
                new_fblock[cj.idx][0] -= 1
                if new_fblock[ci.idx][0] < 0 or  new_fblock[cj.idx][0] < 0:
                    continue
                new_fblock = tuple([(b[0],b[1]) for b in new_fblock])
                print(new_fblock)
                ci_vector[new_fblock] = OrderedDict()
                # beta transfer
                new_fblock = [[b[0],b[1]] for b in ref_fblock]
                new_fblock[ci.idx][1] += 1
                new_fblock[cj.idx][1] -= 1
                if new_fblock[ci.idx][1] < 0 or  new_fblock[cj.idx][1] < 0:
                    continue
                new_fblock = tuple([(b[0],b[1]) for b in new_fblock])
                print(new_fblock)
                ci_vector[new_fblock] = OrderedDict()
                # spin_flip 
                new_fblock = [[b[0],b[1]] for b in ref_fblock]
                new_fblock[ci.idx][0] -= 1
                new_fblock[ci.idx][1] += 1
                new_fblock[cj.idx][0] += 1
                new_fblock[cj.idx][1] -= 1
                if new_fblock[ci.idx][1] < 0 or  new_fblock[cj.idx][1] < 0:
                    continue
                new_fblock = tuple([(b[0],b[1]) for b in new_fblock])
                print(new_fblock)
                ci_vector[new_fblock] = OrderedDict()
            
# create full space for each fock block defined
for fblock,configs in ci_vector.items():
    print("a",fblock,configs)
    dims = []
    for c in ci_vector.clusters:
        # get number of vectors for current fock space
        dims.append(range(c.basis[fblock[c.idx]].shape[1]))
    for newconfig_idx, newconfig in enumerate(itertools.product(*dims)):
        ci_vector[fblock][newconfig] = 0 
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
        #[print(t) for t in terms]
        if fock_li==fock_ri:
            for config_li, config_l in enumerate(configs_l):
                for config_ri, config_r in enumerate(configs_r):        
                    me = 0
                    for term in terms:
                        me_term = term.matrix_element(fock_l,config_l,fock_r,config_r)
                        me += me_term
                    print(me)
        else:
            delta = cp.deepcopy(delta_fock)
            for ci in range(len(clusters)):
                delta[ci][2] = 1 
            for config_li, config_l in enumerate(configs_l):
                for config_ri, config_r in enumerate(configs_r):
                    print(config_l,config_r)
        #clustered_ham.matrix_element(fock_l,fock_r,)


