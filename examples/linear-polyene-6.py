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

np.set_printoptions(suppress=True, precision=7, linewidth=1500)

molecule = '''
C  -6.780165   0.224843   0.000000
C  -5.587871  -0.395931   0.000000
C  -4.310074   0.280869   0.000000
C  -3.110354  -0.354381   0.000000
C  -1.837018   0.309854   0.000000
C  -0.636207  -0.330789   0.000000
C   0.636207   0.330789   0.000000
C   1.837018  -0.309854   0.000000
C   3.110354   0.354381   0.000000
C   4.310074  -0.280869   0.000000
C   5.587871   0.395931   0.000000
C   6.780165  -0.224843   0.000000
H   5.558134   1.485572   0.000000
H  -6.855486   1.309093   0.000000
H   6.855486  -1.309093   0.000000
H  -7.711938  -0.330412   0.000000
H   7.711938   0.330412   0.000000
H  -5.558134  -1.485572   0.000000
H  -4.329470   1.370910   0.000000
H   4.329470  -1.370910   0.000000
H  -3.097480  -1.444853   0.000000
H   3.097480   1.444853   0.000000
H  -1.847524   1.400093   0.000000
H   1.847524  -1.400093   0.000000
H  -0.626779  -1.421145   0.000000
H   0.626779   1.421145   0.000000
'''

cas_nel = 12
cas_norb = 12


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

## Run a CAS-CI calculation for comparison
from pyscf import fci
cisolver = fci.direct_spin1.FCI()
ecas, vcas = cisolver.kernel(h, g, cas_norb, nelec=cas_nel, ecore=ecore,nroots =1,verbose=100)
print("CAS-CI:%10.8f"%(ecas))
print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
eixt()
#print(ecore)



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
blocks = [range(0,4),range(4,8),range(8,12)] # 3 clusters with 2,4,2 orbitals each
init_fspace = ((2, 2),(2, 2),(2, 2))   # Cluster1: (alpha,beta) Cluster2:(alpha,beta) Cluster3:(alpha,beta)

# Initialize the CMF solver. 
oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100,cs_solver=0) #cs_solver,0 for our FCI and 1 for pyscf FCI solver.
oocmf.init() # runs a single step CMF calculation
oocmf.optimize_orbitals()  # optimize the orbitals using gradient
oocmf.form_extra_fspace()  #form excited fock space configurations

clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
ci_vector = oocmf.ci_vector   # lowest energy TPS using the reference Fock space

e_ref = build_hamiltonian_diagonal(clustered_ham,ci_vector) # compute reference energy
print("Reference TPS Energy:%16.10f"%(e_ref+ecore))

# Run the TPSCI calculation on top of the CMF reference 
ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                    pt_type             = 'mp',
                    thresh_cipsi        = 1e-5,
                    thresh_ci_clip      = 1e-6,
                    max_tucker_iter     = 1,
                    nbody_limit         = 4,
                    shared_mem          = 1e9,
                    thresh_search       = 1e-6,
                    thresh_asci         = 1e-2,
                    matvec=4,
                    tucker_state_clip   = 1e-6,
                    tucker_conv_target  = 0,    #converge variational energy
                    nproc               = None)

e2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, etci,
        thresh_asci     = 0,
        thresh_search   = 1e-8,
        pt_type         = 'en',
        matvec          = 4)


tci_dim = len(ci_vector)
ci_vector.print_configs()
print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
print(" TPSCI+PT:       %12.8f             " % (etci+e2+ecore))
print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))

