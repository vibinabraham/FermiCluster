import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import tools 

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)


def test_1():
    np.set_printoptions(suppress=True, precision=2, linewidth=1500)
    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

    ###     PYSCF INPUT
    r0 = 1.2
    molecule = '''
    N   0  0   0
    N   0  0   {}
    '''.format(r0)
    charge = 0
    spin  = 0
    basis_set='6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'

        
    if basis_set == '6-31g':
        cas = True
        cas_nstart = 2
        cas_nstop =  18
        cas_nel = 10
        ###     TPSCI CLUSTER INPUT
        init_fspace = ((2, 2),(1, 1),(1, 1), (1, 1))
        blocks = [range(0,4),range(4,8),range(8,12),range(12,16)]

    if basis_set == 'ccpvdz':
        cas = True
        cas_nstart = 2
        cas_nstop =  28
        cas_nel = 10
        ###     TPSCI CLUSTER INPUT
        init_fspace = ((2, 2),(1, 1),(1, 1), (1, 1), (0, 0))
        blocks = [range(0,4),range(4,10),range(10,16),range(16,22),range(22,26)]

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel

    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    print("Ecore:%16.8f"%ecore)
    C = pmol.C
    K = pmol.K
    mol = pmol.mol
    mo_energy = pmol.mf.mo_energy
    dm_aa = pmol.dm_aa
    dm_bb = pmol.dm_bb

    #clustering for cr (pairing type)
    idx = ordering_diatomics(mol,C,basis_set)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    mo_energy = mo_energy[idx]
    dm_aa = dm_aa[:,idx] 
    dm_aa = dm_aa[idx,:]
    dm_bb = dm_bb[:,idx] 
    dm_bb = dm_bb[idx,:]

    print(dm_aa)
    print(h)

    from pyscf import molden
    #molden.from_mo(pmol.mol, 'h8.molden', C)
    print(h)
    mol = pmol.mol
    if mol.symmetry == True:
        from pyscf import symm
        mo = symm.symmetrize_orb(mol, C)
        osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
        for i in range(len(osym)):
            print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))


    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 0 )


    energy1,t,rmda,rdmb = cmf(clustered_ham, ci_vector, h, g, max_iter=50,dm_guess=(dm_aa,dm_bb),diis=True)
    energy2,t,rdma,rdmb = cmf(clustered_ham, ci_vector, h, g, max_iter=50,dm_guess=None,diis=False)
    assert(np.abs(energy1-energy2)<1e-6)
    
if __name__== "__main__":
    test_1() 
