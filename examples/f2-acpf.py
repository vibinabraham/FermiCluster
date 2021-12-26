import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
from itertools import combinations

import pickle

from tpsci import *
from pyscf_helper import *

import pyscf
from pyscf import gto, scf, ao2mo, molden, lo, mo_mapping, mcscf, cc, mp

ttt = time.time()
np.set_printoptions(suppress=True, precision=3, linewidth=1500)



for ri in range(0,21):
    ###     PYSCF INPUT
    r0 = 1.2 + 0.1*ri
    molecule = '''
    F   0  0   0
    F   0  0   {}
    '''.format(r0)
    charge = 0
    spin  = 0
    basis_set = '6-31g'
    #basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop =  18
    cas_nel = 14


    ###     TPSCI CLUSTER INPUT
    init_fspace = ((2, 2), (1, 1), (2, 2), (2, 2))
    blocks = [range(0,4),range(4,8),range(8,12),range(12,16)]



    #init_fspace = ((3, 3),(2, 2))
    #blocks = [range(0,4),range(4,8)]

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
        mp2 = mp.MP2(mf)
        mp2.frozen = 2
        e_corr, t2 = mp2.kernel()
        mycc = cc.CCSD(mf)
        mycc.frozen = 2
        mycc.kernel()
        print('CCSD correlation energy', mycc.e_corr)
        print('CCSD energy', mycc.e_tot)
        print(' MP2 energy', mp2.e_tot)
        print(' FCI energy',efci)

    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=5e-4,ci_cutoff=5e-4)


    #idx = e1_order(h,1e-2)
    idx = ordering_diatomics(mol,C,basis_set=basis_set)
    norb = cas_nstop - cas_nstart
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    mo_energy = mo_energy[idx]
    dm_aa = dm_aa[:,idx]
    dm_aa = dm_aa[idx,:]
    dm_bb = dm_bb[:,idx]
    dm_bb = dm_bb[idx,:]
    molden.from_mo(mol, 'cas.molden', C)

    print(h)
    print(dm_aa)


    from pyscf import symm
    mo = symm.symmetrize_orb(mol, C)
    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    for i in range(len(osym)):
        print("%4d %8s %16.8f"%(i,osym[i],mo_energy[i]))
    print("r:",r0)
    print(h)

    # Initialize the CMF solver.
    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100,cs_solver=0) #cs_solver,0 for our FCI and 1 for pyscf FCI solver.
    oocmf.init() # runs a single step CMF calculation
    oocmf.optimize_orbitals()  # optimize the orbitals using gradient
    oocmf.form_extra_fspace()  #form excited fock space configurations

    clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
    ci_vector = oocmf.ci_vector   # lowest energy TPS using the given Fock space
    h = oocmf.h
    g = oocmf.g
    C = oocmf.C

    print("core %16.12f"%ecore)

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print("%16.10f"%(edps+ecore))

    ## PT and LCC
    etci = edps

    e2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, etci,
            thresh_asci     = 0,
            thresh_search   = 1e-4,
            shared_mem          = 9e10,
            pt_type         = 'mp',
            nproc = 1,
            matvec          = 4)

    #e2lcc,lcc = pt2infty(clustered_ham,ci_vector,pt_vector,form_H=True,nproc=24)
    print(" E0:          %12.8f      Dim:%6d" % (etci+ecore, len(pt_vector)))
    print(" E cMF+MP2:   %12.8f             " % (etci+e2+ecore))
    #print(" E LCC:       %12.8f             " % (etci+e2lcc+ecore))

    #ecepa = cepa(clustered_ham,ci_vector,pt_vector,'cepa0')

    H00 = build_full_hamiltonian(clustered_ham,ci_vector,iprint=0)
    Hdd = build_full_hamiltonian_parallel2(clustered_ham,pt_vector,iprint=1,nproc=24)
    H0d = build_block_hamiltonian(clustered_ham,ci_vector,pt_vector,iprint=0)

    ecepa0 = cepa_full_ham(H00,Hdd,H0d,ci_vector,pt_vector,'cepa0',cas_nel=cas_nel,cepa_mit=1)
    eacpf1 = cepa_full_ham(H00,Hdd,H0d,ci_vector,pt_vector,'acpf1',cas_nel=cas_nel,cepa_mit=30)
    eaqcc1 = cepa_full_ham(H00,Hdd,H0d,ci_vector,pt_vector,'aqcc1',cas_nel=cas_nel,cepa_mit=30)
    eacpf2 = cepa_full_ham(H00,Hdd,H0d,ci_vector,pt_vector,'acpf2',n_blocks=4,cepa_mit=30)
    eaqcc2 = cepa_full_ham(H00,Hdd,H0d,ci_vector,pt_vector,'aqcc2',n_blocks=4,cepa_mit=30)
    ecisd  = cepa_full_ham(H00,Hdd,H0d,ci_vector,pt_vector,'cisd',cepa_mit=30)
    print(" E CEPA0:%12.8f             " % (etci+ecepa0+ecore))
    print(" E ACPF1:%12.8f             " % (etci+eacpf1+ecore))
    print(" E ACPF2:%12.8f             " % (etci+eacpf2+ecore))
    print(" E AQCC1:%12.8f             " % (etci+eaqcc1+ecore))
    print(" E AQCC2:%12.8f             " % (etci+eaqcc2+ecore))
    print(" E CISD :%12.8f             " % (etci+ecisd+ecore))


