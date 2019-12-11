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

###     PYSCF INPUT
molecule = '''
C
H   1   1.5 
H   1   1.5   2   109
H   1   1.5   3   109     2  120
H   1   1.5   4   109     3 -120
'''
charge = 0
spin  = 0
basis_set = 'sto-3g'

###     TPSCI BASIS INPUT
orb_basis = 'scf'
cas = True
cas_nstart = 1
cas_nstop = 9
cas_nel = 8

###     TPSCI CLUSTER INPUT
blocks = [[0,1],[2,3],[4,5],[6,7]]
blocks = [[0,1,2,3],[4,5,6,7]]
init_fspace = ((1, 1), (1, 1),(1, 1), (1, 1))
init_fspace = ((2, 2), (2, 2))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == sum(nelec))
    nelec = cas_nel


#Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,cas_nstart=1,cas_nstop=9,cas_nel=8,cas=True)

h = pmol.h
g = pmol.g
ecore = pmol.ecore

#cluster using hcore
idx = e1_order(h,cut_off = 1e-1)
h,g = reorder_integrals(idx,h,g)

print(h)

do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=2e-3,ci_cutoff=2e-3)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1e-3,thresh_cipsi=1e-4,max_tucker_iter=20)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
