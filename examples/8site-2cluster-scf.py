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

n_orb = 8
U = 4.
beta = 1.0

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
np.random.seed(2)
#tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
#h += tmp + tmp.T
C = np.eye(h.shape[0])

if 0:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)

blocks = [[0,1,2,3],[4,5,6,7]]
init_fspace = ((4,4),(0,0))
nelec = tuple([sum(x) for x in zip(*init_fspace)])

do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=0)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=0,select_cutoff=1e-3,ci_cutoff=1e-3)

ecore = 0
if do_tci:

    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100)
    oocmf.init()


    h = oocmf.h
    g = oocmf.g
    clustered_ham = oocmf.clustered_ham
    ci_vector = oocmf.ci_vector

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print("%16.10f"%(edps+ecore))

    ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'mp',
                        thresh_cipsi        = 1e-5,
                        thresh_ci_clip      = 1e-6,
                        max_tucker_iter     = 1,
                        nbody_limit         = 4,
                        shared_mem          = 1e9,
                        thresh_search       = 1e-6,
                        thresh_asci         = 1e-2,
                        matvec=3,
                        tucker_state_clip   = None,
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)

    tci_dim = len(ci_vector)

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (etci, tci_dim))
