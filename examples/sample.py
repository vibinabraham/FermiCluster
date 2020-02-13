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

n_orb = 6
U = 1.
beta = 1.0

h, g = get_hubbard_params(n_orb,beta,U,pbc=False)
np.random.seed(2)
#tmp = np.random.rand(h.shape[0],h.shape[1])*0.01
#h += tmp + tmp.T

if 1:
    Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)


blocks = [[0,1,2],[3,4,5]]
init_fspace = ((3,3),(0,0))

blocks = [[0],[1],[2],[3],[4],[5]]
init_fspace = ((1,1),(1,1),(1,1),(0,0),(0,0),(0,0))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
#blocks = [[0,1],[2,3]]


do_fci = 1
do_hci = 0
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=0)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=0,select_cutoff=1e-3,ci_cutoff=1e-3)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=0,max_tucker_iter=0, thresh_cipsi=1e-6, thresh_ci_clip=5e-4)
    tci_dim = len(ci_vector)

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (etci, tci_dim))

