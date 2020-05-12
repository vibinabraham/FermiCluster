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
C      -3.8506247793     0.3491553279     0.0000000000
C      -2.8389399982    -0.5280020896     0.0000000000
C      -1.3948347123    -0.2353721591     0.0000000000
C      -0.8816917679     1.0744159669     0.0000000000
C       0.4942737668     1.3254526739     0.0000000000
C       1.3721153950     0.2260952593     0.0000000000
C       0.9008441494    -1.0905602789     0.0000000000
C      -0.4902543241    -1.3019807863     0.0000000000
C       1.8777861777    -2.1937574239     0.0000000000
C       1.6240316740    -3.5084129326     0.0000000000
C       0.9615521090     2.7226778015     0.0000000000
C       2.2270933749     3.1600692128     0.0000000000
H      -4.8818038425     0.0093128247     0.0000000000
H      -3.7015448436     1.4256972311     0.0000000000
H      -3.0829536800    -1.5905221922     0.0000000000
H      -1.5656111409     1.9186716507     0.0000000000
H       2.4455471010     0.3944973623     0.0000000000
H      -0.8799117961    -2.3163543176     0.0000000000
H       2.9193960034    -1.8720815717     0.0000000000
H       0.6164337399    -3.9151685380     0.0000000000
H       2.4331025434    -4.2324272069     0.0000000000
H       0.1626610523     3.4643997877     0.0000000000
H       3.0839002660     2.4914319609     0.0000000000
H       2.4494335322     4.2227924371     0.0000000000
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

# Run a CAS-CI calculation for comparison
from pyscf import fci
cisolver = fci.direct_spin1.FCI()
ecas, vcas = cisolver.kernel(h, g, cas_norb, nelec=cas_nel, ecore=ecore,nroots =1,verbose=100)
print("CAS-CI:%10.8f"%(ecas))
print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
print(ecore)



## TPSCI 
# Reorder 
mc = mulliken_ordering(mol,h.shape[0],C)
idx = np.where(mc>.9)[1]  #gives index map from atom to local orbital corresponding to that orbital
h,g = reorder_integrals(idx,h,g)
print(h)
C = C[:,idx] # make sure u reorder this too
molden.from_mo(mol, 'cas.molden', C)

# define the orbital blocks and the fock space
blocks = [[0,1,2,3],[4,5,10,11],[6,7,8,9]] 
init_fspace = ((2, 2),(2, 2),(2, 2))  

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
                    thresh_cipsi        = 1e-3,
                    thresh_ci_clip      = 1e-4,
                    max_tucker_iter     = 2,
                    nbody_limit         = 4,
                    shared_mem          = 1e9,
                    thresh_search       = 1e-9,
                    thresh_asci         = 1e-3,
                    matvec=3,
                    tucker_state_clip   = 1e-6,
                    tucker_conv_target  = 0,    #converge variational energy
                    nproc               = None)
ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                    pt_type             = 'en',
                    thresh_cipsi        = 1e-4,
                    thresh_ci_clip      = 1e-5,
                    max_tucker_iter     = 2,
                    nbody_limit         = 4,
                    shared_mem          = 1e9,
                    thresh_search       = 1e-9,
                    thresh_asci         = 1e-3,
                    matvec=3,
                    tucker_state_clip   = 1e-6,
                    tucker_conv_target  = 0,    #converge variational energy
                    nproc               = None)
ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                    pt_type             = 'en',
                    thresh_cipsi        = 1e-5,
                    thresh_ci_clip      = 1e-6,
                    max_tucker_iter     = 2,
                    nbody_limit         = 4,
                    shared_mem          = 1e9,
                    thresh_search       = 0,
                    thresh_asci         = 0,
                    matvec=3,
                    tucker_state_clip   = 1e-6,
                    tucker_conv_target  = 0,    #converge variational energy
                    nproc               = None)
ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                    pt_type             = 'en',
                    thresh_cipsi        = 1e-6,
                    thresh_ci_clip      = 5e-7,
                    max_tucker_iter     = 2,
                    nbody_limit         = 4,
                    shared_mem          = 1e9,
                    thresh_search       = 0,
                    thresh_asci         = 0,
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
