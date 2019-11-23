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


molecule = '''
He      0.00       0.00       0.00
He      0.00       0.00       1.50
He      0.00       0.00       3.00
He      0.00       0.00       4.50
He      0.00       0.00       6.00
'''
charge = 0
spin  = 0
basis = '3-21g'


h,g,ecore = init_pyscf(molecule,charge,spin,basis,local='lowdin')
blocks = [[0,1],[2,3],[4,5],[6,7],[8,9]]
init_fspace = ((1,1),(1,1),(1,1),(1,1),(1,1))
#init_fspace = ((1,1),(1,1))
nelec = tuple([sum(x) for x in zip(*init_fspace)])

print(h)

do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=3e-2,ci_cutoff=1e-2)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1e-3,thresh_cipsi=1e-4,max_tucker_iter=4)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
