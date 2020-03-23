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
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

r0 = 2.00 
r1 = 2.20 
molecule = '''
H   0.1   0   0
H   0.2  {0}  0
H   0   0   {0}
H   0  {0}  {0}
H  {1}  0   0
H  {1} {0}  0.2
H  {1}  0.1   {0}
H  {1} {0}  {0}
'''.format(r0,r1)

r0 = 2.00 
molecule = '''
H   0   0   0
H   0   0   1
H   0   0   2
H   0   0   3
H   0   0   6
H   0   0   7
H   0   0   8
H   0   0   9
'''.format(r0,r1)
charge = 0
spin  = 0
basis_set = 'sto-3g'

###     TPSCI BASIS INPUT
orb_basis = 'lowdin'
cas = False
cas_nel = 8

###     TPSCI CLUSTER INPUT
blocks = [[0,1,2,3],[4,5,6,7]]
init_fspace = ((2, 2), (2, 2))
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
init_fspace = ((2, 2), (2, 2), (2, 2))
blocks = [[0,1],[2,3],[4,5],[6,7]]
init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))

nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == sum(nelec))
    nelec = cas_nel


# Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,cas=False)

h = pmol.h
g = pmol.g
ecore = pmol.ecore

do_fci = 0
do_hci = 0
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

#cluster using hcore
#idx = e1_order(h,cut_off = 2e-1)
#h,g = reorder_integrals(idx,h,g)
if do_tci:
    n_blocks = len(blocks)

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
        e0,tm = cmf(clustered_ham, ci_vector, h, g, max_iter=20,max_nroots=2,dm_guess=None)

    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()
    #ci_vector.add_single_excitonic_states()
    #ci_vector.print_configs()

    precompute_cluster_basis_energies(clustered_ham)
    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print(edps)
    emp2,pt_vector = compute_rspt2_correction(ci_vector, clustered_ham, edps, nproc=1)
    elcc,pt_vector = compute_lcc2_correction(ci_vector, clustered_ham, edps, nproc=1)
    een2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, edps, nproc=1)
    ecepa = cepa(clustered_ham,ci_vector,pt_vector,cepa_shift='cepa0')
    ecisd = cepa(clustered_ham,ci_vector,pt_vector,cepa_shift='cisd')
    ecisd2 = compute_cisd_correction(ci_vector, clustered_ham, nproc=1)
    print(" DPS        :       %12.8f      Dim:%6d" % (edps,1))
    print(" DPS-MPPT2  :       %12.8f      Dim:%6d" % (edps+emp2,len(pt_vector)))
    print(" DPS-ENPT2  :       %12.8f      Dim:%6d" % (edps+een2,len(pt_vector)))
    print(" DPS-CEPA0  :       %12.8f      Dim:%6d" % (edps+ecepa,len(pt_vector)))
    print(" DPS-CISD   :       %12.8f      Dim:%6d" % (edps+ecisd,len(pt_vector)))
    print(" DPS-CISD2  :       %12.8f      Dim:%6d" % (edps+ecisd2,len(pt_vector)))
    print(" DPS        :       %12.8f      Dim:%6d" % (edps+ecore,1))
    print(" DPS-MPPT2  :       %12.8f      Dim:%6d" % (edps+emp2+ecore,len(pt_vector)))
    print(" DPS-ENPT2  :       %12.8f      Dim:%6d" % (edps+een2+ecore,len(pt_vector)))
    print(" DPS-CEPA0  :       %12.8f      Dim:%6d" % (edps+ecepa+ecore,len(pt_vector)))
    print(" DPS-CISD   :       %12.8f      Dim:%6d" % (edps+ecisd+ecore,len(pt_vector)))
    print(" DPS-CISD2  :       %12.8f      Dim:%6d" % (edps+ecisd2+ecore,len(pt_vector)))
    tci_dim = len(pt_vector)
    #pt_vector.clip(.005)
    pt_vector.print_configs()
    edps = edps+ecore
    emp2 = edps+emp2
    een2 = edps+een2
    #ci_vector.normalize()
    #ci_vector.print_configs()
    #print(edps+ecore)
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()
    ci_vector.add_single_excitonic_states()
    H = build_full_hamiltonian(clustered_ham, ci_vector)
    e,v = scipy.sparse.linalg.eigsh(H,10,which='SA')
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    v = v[:,0]
    e0 = e[0]
    e = e[0]
    ci_vector.set_vector(v)
    ci_vector.print_configs()
    print(" FCI        :       %12.8f      Dim:%6d" % (e,len(ci_vector)))
    print(" FCI        :       %12.8f      Dim:%6d" % (e+ecore,len(ci_vector)))
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,selection='cipsi',
        thresh_cipsi=1e-8, thresh_ci_clip=1e-8, max_tucker_iter=4)
    print(" FCI        :       %12.8f      Dim:%6d" % (e+ecore,len(ci_vector)))
    print(" TCI     :       %12.8f      Dim:%6d" % (etci+ecore,1))
    print(" DPS-MPPT2  :       %12.8f      Dim:%6d" % (emp2,len(pt_vector)))
    print(" DPS-ENPT2  :       %12.8f      Dim:%6d" % (een2,len(pt_vector)))
    print(" DPS-CEPA0  :       %12.8f      Dim:%6d" % (edps+ecepa,len(pt_vector)))
    print(" DPS-CISD   :       %12.8f      Dim:%6d" % (edps+ecisd,len(pt_vector)))
    print(" DPS-CISD2  :       %12.8f      Dim:%6d" % (edps+ecisd2,len(pt_vector)))
    print(" FCI        :       %12.8f      Dim:%6d" % (e+ecore,len(ci_vector)))

print("  rad      FCI          Dim          HCI       Dim          CMF      PT Dim       MP2       EN2")
print(" %4.2f  %12.9f   %6d     %12.9f  %6d %12.9f %6d %12.9f %12.9f"%(r0,efci,fci_dim,ehci,hci_dim,edps,tci_dim,emp2,een2))
exit()
