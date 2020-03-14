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


from ci_string import *


pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)
n_cluster_states = 1000

from pyscf import gto, scf, mcscf, ao2mo

r0 = 2.0

molecule= '''
N       0.00       0.00       0.00
N       0.00       0.00       {}'''.format(r0)

charge = 0
spin  = 0
basis_set = '6-31g'

###     TPSCI BASIS INPUT
orb_basis = 'scf'
cas = True
cas_nstart = 2
cas_nstop = 18
cas_nel = 10

###     TPSCI CLUSTER INPUT
blocks = [[0,1,11,15],[2,7,10,14],[3,6,8,13],[4,5,9,12]]
init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))



#Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,
            cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)
h = pmol.h
g = pmol.g
ecore = pmol.ecore
print("Ecore:%16.8f"%ecore)
C = pmol.C
K = pmol.K
mol = pmol.mol
mo_energy = pmol.mf.mo_energy
dm_aa = pmol.dm_aa
dm_bb = pmol.dm_bb

do_fci = 0
do_hci = 0

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,cas_nel,ecore=ecore,select_cutoff=1e-4,ci_cutoff=1e-4)
    print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))

clusters = []
for ci,c in enumerate(blocks):
    clusters.append(Cluster(ci,c))

ci_vector_ref = ClusteredState(clusters)
ci_vector_ref.init(init_fspace)


print(" Clusters:")
[print(ci) for ci in clusters]

clustered_ham = ClusteredOperator(clusters)
print(" Add 1-body terms")
clustered_ham.add_1b_terms(h)
#clustered_ham.add_1b_terms(cp.deepcopy(h))
print(" Add 2-body terms")
clustered_ham.add_2b_terms(g)
#clustered_ham.add_2b_terms(cp.deepcopy(g))
#clustered_ham.combine_common_terms(iprint=1)


do_cmf = 1
if do_cmf:
    # Get CMF reference
    #cmf(clustered_ham, ci_vector, cp.deepcopy(h), cp.deepcopy(g), max_iter=4)
    #cmf(clustered_ham, ci_vector, h, g, max_iter=2)
    #cmf(clustered_ham, ci_vector, h, g, max_iter=50,max_nroots=50,dm_guess=(dm_aa,dm_bb),diis=True)
    cmf(clustered_ham, ci_vector_ref, h, g, max_iter=50,max_nroots=1000,dm_guess=(dm_aa,dm_bb),diis=True)
else:
    print(" Build cluster basis and operators")
    for ci_idx, ci in enumerate(clusters):
        ci.form_eigbasis_from_ints(h,g)
        #ci.form_eigbasis_from_ints(h,g,max_roots=1)
    
        print(" Build new operators for cluster ",ci.idx)
        ci.build_op_matrices()

ci_vector = ci_vector_ref.copy()
e0 = build_hamiltonian_diagonal(clustered_ham,ci_vector)
print(" Energy of reference TPS: %12.8f (elec)"%(e0))
print(" Energy of reference TPS: %12.8f (total)"%(e0+ecore))


if 0:
    e2, pt_vector = compute_pt2_correction(ci_vector, clustered_ham, e0, nproc=4)
    print(" Energy of TPS(2): %12.8f (elec)"%(e0+e2))
    print(" Energy of TPS(2): %12.8f (total)"%(e0+e2+ecore))
    
    ci_vector.add(pt_vector)
    ci_vector.normalize()
    
    hosvd(ci_vector, clustered_ham, trim=None, hshift=None)


ci_vector = ClusteredState(clusters)
ci_vector.init(init_fspace)

    
ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-4,
            thresh_ci_clip=1e-5, max_tucker_iter = 4, hshift=1e-8, thresh_asci=1e-1 )

ci_vector.add(pt_vector)
ci_vector.normalize()

hosvd(ci_vector, clustered_ham, trim=1e-6)


clusters, clustered_ham = join(clusters,0,1,h,g)   # (0,1)(2)(3)
clusters, clustered_ham = join(clusters,1,2,h,g)   # (2,3)(0,1)


ci_vector_ref = ClusteredState(clusters)
#ci_vector.init( ((3, 3), (1, 1), (1, 1)) )
ci_vector_ref.init( ((2, 2), (3, 3)) )

ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector_ref.copy(), clustered_ham, thresh_cipsi=1e-4,
            thresh_ci_clip=1e-5, max_tucker_iter = 1, hshift=None, thresh_asci=1e-1)

ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector_ref.copy(), clustered_ham, thresh_cipsi=1e-6,
            thresh_ci_clip=1e-7, max_tucker_iter = 5, hshift=None, thresh_asci=1e-2 )


e2, pt_vector = compute_pt2_correction(ci_vector, clustered_ham, e0)


    
