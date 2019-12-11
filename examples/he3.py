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
He      0.00       0.00       0.00
He      0.00       0.00       1.50
He      0.20       1.50       0.00
'''
charge = 0
spin  = 0
basis_set = '3-21g'

###     TPSCI BASIS INPUT
orb_basis = 'lowdin'
cas = False
#cas_nstart = 2
#cas_nstop = 10
#cas_nel = 10

###     TPSCI CLUSTER INPUT
blocks = [[0,1],[2,3],[4,5]]
init_fspace = ((1, 1), (1, 1), (1, 1))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == nelec)
    nelec = cas_nel


#Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis)

h = pmol.h
g = pmol.g
ecore = pmol.ecore

##cluster using hcore
#idx = e1_order(h,cut_off = 1e-1)
#h,g = reorder_integrals(idx,h,g)


do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=2e-3,ci_cutoff=2e-3)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1e-4,thresh_cipsi=1e-7,max_tucker_iter=20)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
