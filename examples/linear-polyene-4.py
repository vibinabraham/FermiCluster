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
from pyscf import gto, scf, ao2mo, molden, lo, mo_mapping, mcscf
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

np.set_printoptions(suppress=True, precision=3, linewidth=1500)

molecule = '''
C  -4.308669   0.197146   0.000000
C  -3.110874  -0.411353   0.000000
C  -1.839087   0.279751   0.000000
C  -0.634371  -0.341144   0.000000
C   0.634371   0.341144   0.000000
C   1.839087  -0.279751   0.000000
C   3.110874   0.411353   0.000000
C   4.308669  -0.197146   0.000000
H  -4.394907   1.280613   0.000000
H   4.394907  -1.280613   0.000000
H  -5.234940  -0.367304   0.000000
H   5.234940   0.367304   0.000000
H  -3.069439  -1.500574   0.000000
H   3.069439   1.500574   0.000000
H  -1.871161   1.369551   0.000000
H   1.871161  -1.369551   0.000000
H  -0.607249  -1.431263   0.000000
H   0.607249   1.431263   0.000000
'''

cas_nel = 8
cas_norb = 8


#PYSCF inputs
mol = gto.Mole(atom=molecule,
    symmetry = True,basis = 'sto-3g' )
mol.build()
print("symmertry: ",mol.topgroup)

#SCF 
mf = scf.RHF(mol)
mf.verbose = 4
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-9
mf.run(max_cycle=200)


## Active space selection


h,ecore,g,C = get_pi_space(mol,mf,cas_norb,cas_nel,local=True)

# Run a CAS-CI calculation for comparison
from pyscf import fci
cisolver = fci.direct_spin1.FCI()
ecas, vcas = cisolver.kernel(h, g, cas_norb, nelec=cas_nel, ecore=ecore,nroots =1,verbose=100)
print("CAS-CI:%10.8f"%(ecas))
print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
print(ecore)



## TPSCI 

# Reorder the orbitals for TPSCI. (optional, if u know exact cluster u are looking for)
# U can use mulliken order to reorder ur orbitals in same order as the C atoms.
# For this example, we can just use the simpler reordering since its a 1D system

#mc = mulliken_ordering(mol,h.shape[0],C)
#idx = np.where(mc>.9)[1]  #gives index map from atom to local orbital corresponding to that orbital
idx = e1_order(h,1e-1)  #this function works for 1d systems only. its better to know the atom ordering of the atoms

# Reorder 
h,g = reorder_integrals(idx,h,g)
print(h)
C = C[:,idx] # make sure u reorder this too
molden.from_mo(mol, 'cas.molden', C)

# define the orbital blocks and the fock space you wish to initialize.
blocks = [range(0,2),range(2,6),range(6,8)] # 3 clusters with 2,4,2 orbitals each
init_fspace = ((1, 1),(2, 2),(1, 1))   # Cluster1: (alpha,beta) Cluster2:(alpha,beta) Cluster3:(alpha,beta)

# Initialize the CMF solver. 
oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100)
oocmf.init() # runs a single step CMF calculation

clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
ci_vector = oocmf.ci_vector   # lowest energy TPS using the given Fock space

e_ref = build_hamiltonian_diagonal(clustered_ham,ci_vector) # compute reference energy
print("Reference TPS Energy:%16.10f"%(e_ref+ecore))

# Run the TPSCI calculation on top of the CMF reference 
ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                    pt_type             = 'en',
                    thresh_cipsi        = 1e-6,
                    thresh_ci_clip      = 1e-7,
                    max_tucker_iter     = 0,
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
print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
print(" TPSCI+PT:       %12.8f             " % (etci+e2+ecore))
print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
