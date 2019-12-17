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
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

###     PYSCF INPUT
r0 = 1.0 
molecule = '''
N      0.00       0.00       0.00
N      0.00       0.00       {}'''.format(r0)
charge = 0
spin  = 0
basis_set = 'sto-3g'

###     TPSCI BASIS INPUT
orb_basis = 'scf'
cas = True
cas_nstart = 2
cas_nstop = 10
cas_nel = 10

###     TPSCI CLUSTER INPUT
blocks = [[0,1,2,3],[4,5,6,7]]
init_fspace = ((3, 3), (2, 2))

#Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,
            cas,cas_nstart,cas_nstop,cas_nel)

h = pmol.h
g = pmol.g
ecore = pmol.ecore

do_fci = 1
do_hci = 1
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,cas_nel,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)
#cluster using hcore
idx = e1_order(h,cut_off = 1e-4)
h,g = reorder_integrals(idx,h,g)
if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1,thresh_cipsi=1,max_tucker_iter=0,max_cipsi_iter=1)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


print("  rad      FCI          Dim          HCI       Dim          TPSCI      Dim       TPSCI(2)")
print(" %4.2f  %12.9f   %6d     %12.9f  %6d %12.9f %6d %12.9f"%(r0,efci,fci_dim,ehci,hci_dim,etci,tci_dim,etci2))

