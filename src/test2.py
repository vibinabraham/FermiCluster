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

n_orb = 8
U = 1
beta = 1.0

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
np.random.seed(2)
tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
h += tmp + tmp.T

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

cipsi_thresh = 1e-4

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
blocks = [[0],[1],[2],[3],[4],[5],[6],[7]]
blocks = [[0,1,2,3,4],[5,6,7]]
#blocks = [[0],[1],[2,3,4,5],[6,7]]
n_blocks = len(blocks)
clusters = []

for ci,c in enumerate(blocks):
    clusters.append(Cluster(ci,c))

ci_vector = ClusteredState(clusters)
#ci_vector.init(((2,2),(0,0)))
#ci_vector.init(((2,2),(2,2),(0,0)))
#ci_vector.init(((1,1),(1,1),(1,1),(1,1),(0,0),(0,0),(0,0),(0,0)))
#ci_vector.init(((2,2),(2,2),(0,0),(0,0)))
#ci_vector.init(((2,2),(2,2),(0,0)))
ci_vector.init(((4,4),(0,0)))

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

pt_vector = ci_vector.copy()
for it in range(4):
    print(" Build full Hamiltonian")
    H = build_full_hamiltonian(clustered_ham, ci_vector)

    print(" Diagonalize Hamiltonian Matrix:")
    e,v = np.linalg.eigh(H)
    idx = e.argsort()   
    e = e[idx]
    v = v[:,idx]
    v0 = v[:,0]
    e0 = e[0]
    print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))
   
    ci_vector.zero()
    ci_vector.set_vector(v0)
    pt_vector = matvec1(clustered_ham, ci_vector)
    pt_vector.print()
  
    var = pt_vector.norm() - e0*e0
    print(" Variance: %12.8f" % var)
    
    
    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]

    
    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])

    print(" Norm of CI vector = %12.8f" %ci_vector.norm())
    print(" Dimension of CI space: ", len(ci_vector)) 
    print(" Dimension of PT space: ", len(pt_vector)) 
    print(" Compute Denominator")
    #next_ci_vector = cp.deepcopy(ci_vector) 
    # compute diagonal for PT2
    denom = 1/(e0 - build_hamiltonian_diagonal(clustered_ham, pt_vector))
    pt_vector_v = pt_vector.get_vector()
    pt_vector_v.shape = (pt_vector_v.shape[0])
    
    e2 = np.multiply(denom,pt_vector_v)
    pt_vector.set_vector(e2)
    e2 = np.dot(pt_vector_v,e2)
    
    print(" PT2 Energy Correction = %12.8f" %e2)
    print(" PT2 Energy Total      = %12.8f" %(e0+e2))

    print(" Choose which states to add to CI space")

    for fockspace,configs in pt_vector.items():
        for config,coeff in configs.items():
            if coeff*coeff > cipsi_thresh:
                if fockspace in ci_vector:
                    ci_vector[fockspace][config] = 0
                else:
                    ci_vector.add_fockblock(fockspace)
                    ci_vector[fockspace][config] = 0
    print(" Next iteration CI space dimension", len(ci_vector))

