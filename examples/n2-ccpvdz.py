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


# set memory requirements
numpy_memory = 2
###     PYSCF INPUT
r0 = 1.0975
#r0 = 0.9 + 0.1 * ri
molecule = '''
N   0  0   0
N   0  0   {}
'''.format(r0)
charge = 0
spin  = 0
basis_set = 'ccpvdz'

###     TPSCI BASIS INPUT
orb_basis = 'scf'
cas = True
cas_nstart = 2
cas_nstop =  28
cas_nel = 10

###     TPSCI CLUSTER INPUT
init_fspace = ((2, 2),(1, 1),(1, 1), (1, 1), (0, 0))
blocks = [range(0,6),range(6,10),range(10,16),range(16,22),range(22,26)]

nelec = tuple([sum(x) for x in zip(*init_fspace)])
if cas == True:
    assert(cas_nel == sum(nelec))
    nelec = cas_nel


# Integrals from pyscf
pmol = PyscfHelper()
pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True)
                #loc_nstart=loc_start,loc_nstop = loc_stop)

C = pmol.C
h = pmol.h
g = pmol.g
ecore = pmol.ecore
print("Ecore:%16.8f"%ecore)
mol = pmol.mol
mf = pmol.mf
mo_energy = mf.mo_energy[cas_nstart:cas_nstop]
dm_aa = pmol.dm_aa
dm_bb = pmol.dm_bb

do_fci = 0
do_hci = 0
do_tci = 1

if do_fci:
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
if do_hci:
    ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

idx = ordering_diatomics(mol,C)
h,g = reorder_integrals(idx,h,g)
C = C[:,idx]
mo_energy = mo_energy[idx]
dm_aa = dm_aa[:,idx] 
dm_aa = dm_aa[idx,:]
dm_bb = dm_bb[:,idx] 
dm_bb = dm_bb[idx,:]

print(h)
print(dm_aa)

from pyscf import symm
mo = symm.symmetrize_orb(mol, C)
osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
#symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
for i in range(len(osym)):
    print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))
print("r:",r0)
print(h)
if do_tci:
    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)


    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.combine_common_terms(iprint=1)


    do_cmf = 1
    if do_cmf:
        # Get CMF reference
        cmf(clustered_ham, ci_vector, h, g, max_iter=20,max_nroots=2,dm_guess=(dm_aa,dm_bb))

    else:
        # Get vaccum reference
        for ci_idx, ci in enumerate(clusters):
            print()
            print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
            ci.form_eigbasis_from_ints(h,g,max_roots=50)
            print(" Build these local operators")
            print(" Build mats for cluster ",ci.idx)
            ci.build_op_matrices()

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print("CMF Energy: %16.8f"%(edps[0]+ecore))
    exit()

    ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, thresh_cipsi=1e-6,
            thresh_ci_clip=5e-4, max_tucker_iter = 2,hshift=None,nproc=1)
    print("TPSCI:    %14.8f"%(e0+ecore))
    print("TPSCI(2): %14.8f"%(e2+ecore))

    print(edps)
    ci_vector.print_configs()

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
    print("%6.3f %16.8f   %16.8f  %16.8f  %16.8f"%(r0,Escf,Edps,Ecmf,efci))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))

