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
C = pmol.C

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
    # Initialize the CMF solver. 
    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100)
    oocmf.init() # runs a single step CMF calculation
    #oocmf.optimize_orbitals()  # optimize the orbitals using gradient
    oocmf.form_extra_fspace()  #form excited fock space configurations

    clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
    ci_vector = oocmf.ci_vector   # lowest energy TPS using the given Fock space

    e_ref = build_hamiltonian_diagonal(clustered_ham,ci_vector) # compute reference energy
    print("Reference TPS Energy:%16.10f"%(e_ref+ecore))

    # Run the TPSCI calculation on top of the CMF reference 
    ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'en',
                        thresh_cipsi        = 1e-6,
                        thresh_ci_clip      = 1e-7,
                        max_tucker_iter     = 3,
                        nbody_limit         = 4,
                        shared_mem          = 1e9,
                        thresh_search       = 1e-6,
                        thresh_asci         = 1e-2,
                        matvec=3,
                        tucker_state_clip   = 1e-6,
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)

    e2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, etci,
            thresh_asci     = 0,
            thresh_search   = 1e-8,
            pt_type         = 'en',
            matvec          = 3)
    tci_dim = len(ci_vector)
    ci_vector.print_configs()
    etci = etci+ecore


print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))
print(" HCI:        %12.9f Dim:%6d"%(ehci,hci_dim))
print(" FCI:        %12.9f Dim:%6d"%(efci,fci_dim))
