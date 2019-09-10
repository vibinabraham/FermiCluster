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

ttt = time.time()

n_orb = 8
U = 1.
beta = 1.

h, g = get_hubbard_params(n_orb,beta,U)

if 0:
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

clustered_ham.add_ops_to_clusters()
print(" Build these local operators")
for c in clusters:
    print(c)
    print(c.ops)

for ci_idx, ci in enumerate(clusters):
    assert(ci_idx == ci.idx)
    print(" Extract local operator for cluster",ci.idx)
    opi = clustered_ham.extract_local_operator(ci_idx)
    [print(t) for t in opi.terms]
