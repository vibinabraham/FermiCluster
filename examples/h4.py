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
ttt = time.time()
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues



###     PYSCF INPUT
r0 = 1.23 
r1 = 1.50 
molecule = '''
H      -{1}       0.00       0.00
H      -{1}       0.00       {0}
H      0.00       0.00       0.00
H      0.00       0.00       {0}
H       {1}       0.00       0.00
H       {1}       0.00       {0}'''.format(r0,r1)
charge = 0
spin  = 0
basis_set = 'sto-3g'

###     TPSCI BASIS INPUT
orb_basis = 'lowdin'
cas = False
cas_nstart = 0
cas_nstop = 6
cas_nel = 6
fock = (3,3)

norb = cas_nstop - cas_nstart
cas_norb = cas_nstop - cas_nstart

###     TPSCI CLUSTER INPUT
blocks = [range(4)]
init_fspace = ((2, 2),)

blocks = [range(2),range(2,4),range(4,6)]
init_fspace = ((1, 1),(1, 1), (1, 1))




nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == sum(nelec))
    nelec = cas_nel


# Integrals from pyscf
#Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,cas=False,
                cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel)
                #loc_nstart=loc_start,loc_nstop = loc_stop)

C = pmol.C
h = pmol.h
g = pmol.g
ecore = pmol.ecore
print("Ecore:%16.8f"%ecore)
mol = pmol.mol
mf = pmol.mf
mo_energy = mf.mo_energy[cas_nstart:cas_nstop]




from pyscf import symm
mo = symm.symmetrize_orb(mol, C)
osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
##symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
for i in range(len(osym)):
    print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))
from pyscf import molden
molden.from_mo(mol, 'cas.molden', C)

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
                    thresh_ci_clip      = 1e-7,
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

from pyscf import fci
cisolver = fci.direct_spin1.FCI()
ecas, vcas = cisolver.kernel(h, g, cas_norb, nelec=cas_nel, ecore=ecore,nroots =1,verbose=100)

print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
print(" TPSCI+PT:       %12.8f             " % (etci+e2+ecore))
print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
