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

dim_a = 4
dim_b = 4
U = 1
beta1 = 1
beta2 = 0.2
n_orb = dim_a * dim_b

h, g = make_2d_lattice(dim_a,dim_b,beta1,beta2,U)


blocks = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
init_fspace = ((2,2),(2,2),(2,2),(2,2))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
#blocks = [[0,1],[2,3]]


do_fci = 0
do_hci = 1
do_tci = 0

if do_fci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=0)
if do_hci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    Escf,orb,h2,g2,C = run_hubbard_scf(h,g,n_orb//2)
    ehci, hci_dim = run_hci_pyscf(h2,g2,nelec,ecore=0,select_cutoff=1e-3,ci_cutoff=1e-3)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=0,
        thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=0,max_cipsi_iter=20)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (etci, tci_dim))

