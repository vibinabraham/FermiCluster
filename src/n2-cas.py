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
np.set_printoptions(suppress=True, precision=3, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)


# set memory requirements
numpy_memory = 2

###     PYSCF INPUT
r0 = 2.0977
molecule = '''
N      0.00       0.00       0.00
N      0.00       0.00       {}'''.format(r0)

charge = 0
spin  = 0
basis_set = 'ccpvdz'

###     TPSCI BASIS INPUT 
orb_basis = 'scf'
cas = True
cas_nstart = 2
cas_nstop = 28
cas_nel = 10
loc_nstart = 2
loc_nstop = 10

###   TPSCI CLUSTER INPUT
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10],[11,12,13],[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]]
init_fspace = ((3,3),(2,2),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0))

blocks = [[0,1],[2,3,4,5,6,7],[8,9,10,11,12,13],[14,15,16,25],[17,18,23,24],[19,20,21,22]]
init_fspace = ((2,2),(3,3),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0))


pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,cas,cas_nstart,cas_nstop,cas_nel,loc_nstart,loc_nstop)

do_tci = 1

print(pmol.h.shape)

idx = ordering(pmol,cas,cas_nstart,cas_nstop,loc_nstart,loc_nstop,ordering='hcore')
h,g = reorder_integrals(idx,pmol.h,pmol.g)
ecore = pmol.ecore
print(" Ecore: %18.12f" %ecore)

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


# Get CMF reference
cmf(clustered_ham, ci_vector, h, g, max_iter=10, max_nroots=100)


ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-5,
        thresh_ci_clip=1e-4, max_tucker_iter = 20, search_max_nbody=2)

