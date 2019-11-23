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
N      0.00       0.00       0.00
N      0.00       0.00       0.90
'''
charge = 0
spin  = 0
basis_set = 'ccpvdz'

###     TPSCI BASIS INPUT
orb_basis = 'boys'
cas = True
cas_nstart = 2
cas_nstop = 10
cas_nel = 10

###     TPSCI CLUSTER INPUT
blocks = [[0,1,2,3],[4,5],[6,7]]
init_fspace = ((3, 3), (1, 1), (1, 1))



#Integrals from pyscf
h,g,ecore = init_pyscf(molecule,charge,spin,basis_set,
                        orb_basis,cas,cas_nstart,cas_nstop, cas_nel)

#cluster using hcore
idx = e1_order(h,cut_off = 1e-4)
h,g = reorder_integrals(idx,h,g)


do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,cas_nel,ecore=ecore,select_cutoff=1e-4,ci_cutoff=1e-4)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1e-5,thresh_cipsi=1e-7,max_tucker_iter=20)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
