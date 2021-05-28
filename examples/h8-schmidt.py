import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

import pickle

from tpsci import *
from pyscf_helper import *
from pyscf import gto, scf,fci, mcscf
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=6, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)


###     PYSCF INPUT
r0 = 2.0
molecule = '''
H  0 0 0.1
H 0 1 -1
H 0 1  1
H 0 2  0
H 0 4  0
H 0 5 -1
H 0 5  1
H 0 6  0
'''

charge = 0
spin  = 0
basis_set = 'sto-3g'
#basis_set = 'sto-3g'
###     TPSCI BASIS INPUT
orb_basis = 'lowdin'
cas = False
cas_nstart = 0
cas_nstop = 8
cas_nel = 8
fock = (4,4)

norb = cas_nstop - cas_nstart
cas_norb = cas_nstop - cas_nstart

###     TPSCI CLUSTER INPUT

blocks = [range(2),range(2,4),range(4,6)]
init_fspace = ((1, 1), (1, 1), (1, 1))

blocks = [range(4),range(4,8)]
init_fspace = ((2, 2),(2,2))



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
oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=1,cs_solver=0) #cs_solver,0 for our FCI and 1 for pyscf FCI solver.
oocmf.init() # runs a single step CMF calculation
oocmf.optimize_orbitals()  # optimize the orbitals using gradient
oocmf.form_extra_fspace()  #form excited fock space configurations

ci_vector = oocmf.ci_vector
clustered_ham = oocmf.clustered_ham
clusters = clustered_ham.clusters


h = oocmf.h
g = oocmf.g


print(" Ecore   :%16.8f"%ecore)

if 1:
    n_blocks = len(blocks)
    clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    ci_vector = ClusteredState()
    ci_vector.init(clusters, init_fspace)

    ecmf, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, diis = True,dm_guess    = None,max_iter=20)

    print(rdm_a)
    for ci in clusters:
        ci.form_schmidt_basis(h,g,rdm_a,rdm_b, thresh_schmidt=5e-3)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)
        ci.build_effective_cmf_hamiltonian(h,g,rdm_a,rdm_b)

    #for ci in clusters:
    #    #print(ci.basis)
    #    for key in ci.basis.keys():
    #        print(key)
    #        temp = np.load("fock_"+str(ci.idx+1)+"_"+str(key))
    #        #print(ci.basis[key])
    #        #print(temp)
    #        print(temp.T @ ci.basis[key])


ci_vector.expand_to_full_space(clusters)
print(len(ci_vector))
H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)
e,v = np.linalg.eigh(H)
idx = e.argsort()
e = e[idx]
v = v[:,idx]
print(e)
print(H)

for i in range(e.shape[0]):
    print("%12.8f"%e[i])
