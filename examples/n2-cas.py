import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from tpsci import *
from PyscfHelper import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=3, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)


# set memory requirements
numpy_memory = 2

###     PYSCF INPUT
r0 = 1.5
molecule = '''
N      0.00       0.00       0.00
N      0.00       0.00       {}'''.format(r0)

charge = 0
spin  = 0
basis_set = 'ccpvdz'

###     TPSCI BASIS INPUT 
orb_basis = 'scf'
cas = True
cas_nstart = 2
cas_nstop = 28
cas_nel = 10
loc_nstart = 2
loc_nstop = 10

###   TPSCI CLUSTER INPUT
blocks = [[0,1,2,3],[4,5,6,7],[8,9,10],[11,12,13],[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]]
init_fspace = ((3,3),(2,2),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0))


pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,cas,cas_nstart,cas_nstop,cas_nel,loc_nstart,loc_nstop)

do_tci = 1

print("pmol has h")
print(pmol.h.shape)

idx = ordering(pmol,cas,cas_nstart,cas_nstop,loc_nstart,loc_nstop,ordering='hcore')
h,g = reorder_integrals(idx,pmol.h,pmol.g)
ecore = pmol.ecore
print(h)

if do_tci:
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
        thresh_ci_clip=1e-1,thresh_cipsi=1e-1,max_tucker_iter=0,max_cipsi_iter=1)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)
    print(etci)

