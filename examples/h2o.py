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
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

###     PYSCF INPUT
molecule = '''
O
H   1   2.1 
H   1   2.1   2   106
'''
charge = 0
spin  = 0
basis_set = '6-31g'

###     TPSCI BASIS INPUT
orb_basis = 'ibmo'
cas = True
cas_nstart = 1
cas_nstop =  11
loc_start = 1
loc_stop = 7
cas_nel = 8

###     TPSCI CLUSTER INPUT
blocks = [[0,1],[2,3],[4,5],[6,7]]
blocks = [[0,1],[2,3],[4,5],[6,7]]
blocks = [[0,1,2],[3,4],[5,6]]
blocks = [[0,1],[2,4],[3,5],[6,7],[8,9]]

init_fspace = ((1, 1), (1, 1),(1, 1), (1, 1))
init_fspace = ((2, 2), (2, 2))
init_fspace = ((2, 2), (1, 1),(1,1), (0, 0), (0, 0))


#if orb_basis == 'scf':
#    blocks = [[0,1],[2,3,4,5]]
#    init_fspace = ((1, 1), (3, 3))

nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == sum(nelec))
    nelec = cas_nel


# Integrals from pyscf
h,g,ecore = init_pyscf(molecule,charge,spin,basis_set,orb_basis,cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,loc_nstart=loc_start,loc_nstop = loc_stop)

#cluster using hcore
idx = e1_order(h,cut_off = 0.2)
h,g = reorder_integrals(idx,h,g)
print(h)
print(idx)
exit()

do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=5)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
