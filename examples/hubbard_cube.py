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
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

np.set_printoptions(suppress=True, precision=3, linewidth=1500)

dim_a = 3 #n cluster
dim_b = 4 # site per cluster
U = 1
beta1 = 1
beta2 = 1/(2**6)
print(beta2)
n_orb = dim_a * dim_b

h, g = make_stack_lattice(dim_a,dim_b,beta1,beta2,U,pbc = True)


blocks = [range(4),range(4,8),range(8,12)]
init_fspace = ((2,2),(2,2),(2,2))
nelec = tuple([sum(x) for x in zip(*init_fspace)])


do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    Escf,orb,h2,g2,C = run_hubbard_scf(h,g,n_orb//2)
    efci, fci_dim = run_fci_pyscf(h2,g2,nelec,ecore=0)
if do_hci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    Escf,orb,h2,g2,C = run_hubbard_scf(h,g,n_orb//2)
    ehci, hci_dim = run_hci_pyscf(h2,g2,nelec,ecore=0,select_cutoff=1e-3,ci_cutoff=1e-3)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=0,
        thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=10,max_cipsi_iter=20)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (etci, tci_dim))

