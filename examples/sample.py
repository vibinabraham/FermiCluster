import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *
from bc_cipsi import *
import pyscf
ttt = time.time()

n_orb = 6
U = 5.
beta = 1.0

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
np.random.seed(2)
tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
h += tmp + tmp.T

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)


blocks = [[0,1,2],[3,4,5]]
init_fspace = ((3,3,),(0,0))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
#blocks = [[0,1],[2,3]]

def run_tpsci(h,g,blocks,init_fspace,ecore=0):
# {{{
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

    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()

    ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-5, thresh_ci_clip=1e-5,max_iter=20)
    return ci_vector, pt_vector, e0, e2 
# }}}

do_fci = 1
do_hci = 1
do_tci = 1
do_cas = 0

if do_fci:
    # FCI
    from pyscf import fci
    efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=0, verbose=5)
    fci_dim = ci.shape[0] * ci.shape[1]
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))

if do_cas:
    mc = mcscf.CASCI(myhf, nelec, h.shape[0])
    mc.kernel(verbose=100)
    ecas = mc.e_tot 
    casdim = mc.ci.shape[0] * mc.ci.shape[1]
    print(casdim)

if do_hci:
    #heat bath ci
    from pyscf import mcscf
    from pyscf.hci import hci
    cisolver = hci.SCI()
    cisolver.select_cutoff = 1e-3
    cisolver.ci_coeff_cutoff = 1e-3
    ehci, civec = cisolver.kernel(h,g,h.shape[1],nelec,ecore=0,verbose=100)
    hci_dim = civec[0].shape[0]
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))

if do_tci:
    ci_vector, pt_vector, e0, e2 = run_tpsci(h,g,blocks,init_fspace,ecore=0)
    tci_dim = len(ci_vector) 
    print(" TPSCI:        %12.8f Dim:%6d" % (e0, tci_dim))

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_cas:
    print(casdim)
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (e0, tci_dim))

