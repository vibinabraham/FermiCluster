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
np.set_printoptions(suppress=True, precision=3, linewidth=1500)

molecule = '''
6  -2.433661   0.708302   0.000000
6  -2.433661  -0.708302   0.000000
1  -3.378045  -1.245972   0.000000
1  -3.378045   1.245972   0.000000
6  -1.244629   1.402481   0.000000
6  -1.244629  -1.402481   0.000000
6  -0.000077   0.717168   0.000000
6  -0.000077  -0.717168   0.000000
1  -1.242734   2.490258   0.000000
1  -1.242734  -2.490258   0.000000
6   1.244779   1.402533   0.000000
6   1.244779  -1.402533   0.000000
6   2.433606   0.708405   0.000000
6   2.433606  -0.708405   0.000000
1   1.242448   2.490302   0.000000
1   1.242448  -2.490302   0.000000
1   3.378224   1.245662   0.000000
1   3.378224  -1.245662   0.000000
'''

cas_nel = 10
cas_norb = 10


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


do_fci = 1
if do_fci:
    # Run a CAS-CI calculation for comparison
    from pyscf import fci
    cisolver = fci.direct_spin1.FCI()
    ecas, vcas = cisolver.kernel(h, g, cas_norb, nelec=cas_nel, ecore=ecore,nroots =1,verbose=100)
    print("CAS-CI:%10.8f"%(ecas))
    print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
    print(ecore)

print(C.shape[0])
print(C.shape[1])
## TPSCI 
mc = mulliken_ordering(mol,h.shape[0],C)
print(mc.shape)
exit()
idx = np.where(mc>.9)[1]  #gives index map from atom to local orbital corresponding to that orbital

# Reorder 
h,g = reorder_integrals(idx,h,g)
print(h)
C = C[:,idx] # make sure u reorder this too
molden.from_mo(mol, 'cas.molden', C)

# define the orbital blocks and the fock space you wish to initialize.
blocks = [range(0,5),range(5,10)] # 3 clusters with 2,4,2 orbitals each
init_fspace = ((2, 3),(3, 2))   # Cluster1: (alpha,beta) Cluster2:(alpha,beta) Cluster3:(alpha,beta)

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
