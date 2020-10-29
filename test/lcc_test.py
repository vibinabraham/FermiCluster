
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from fermicluster import *
from pyscf_helper import *

import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues


def test_1():
    ###     PYSCF INPUT
    r0 = 1.60
    molecule = '''
    H     -1.00       0.00       2.00
    H      0.00       0.00       2.00
    H      0.00       0.00       0.00
    H      1.23       0.00       0.00
    H      1.23       0.00       {0}
    H      0.00       0.00       {0}'''.format(r0)

    charge = 0
    spin  = 0
    basis_set = '6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    cas_nstart = 0
    cas_nstop = 12
    cas_nel = 6

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    init_fspace = ((1, 1), (1, 1),(1,1))



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
    molden.from_mo(mol, 'h8.molden', C)



    # Initialize the CMF solver. 
    n_blocks = len(blocks)
    clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]

    print(" Ecore   :%16.8f"%ecore)
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


    Ecmf, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter = 20)
    ecmf = Ecmf + ecore

    for ci_idx, ci in enumerate(clusters):
        ci.form_fockspace_eigbasis(h, g, [init_fspace[ci_idx]], max_roots=5, rdm1_a=rdm_a, rdm1_b=rdm_b,iprint=1)
        #fspaces_i = ci.possible_fockspaces()
        #ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=3, rdm1_a=rdm_a, rdm1_b=rdm_b,iprint=1)

        print(" Build new operators for cluster ",ci.idx)
        ci.build_op_matrices(iprint=0)
        ci.build_local_terms(h,g)


    emp2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, Ecmf,
            thresh_asci     = 0,
            thresh_search   = 1e-9,
            pt_type         = 'mp',
            nbody_limit = 4,
            matvec          = 4)


        
    H = build_full_hamiltonian(clustered_ham,pt_vector,iprint=0, opt_einsum=True)
    np.fill_diagonal(H,0)

    ci = pt_vector.get_vector()

    sigma_vec = build_sigma(clustered_ham,pt_vector,iprint=0, opt_einsum=True)
    sigma = sigma_vec.get_vector()

    sigma_ref = H@ ci
    for i in range(sigma.shape[0]):
        print("%16.9f"%(sigma_ref[i] - sigma[i]))
        assert(abs(sigma_ref[i] - sigma[i]) < 1e-12)

    elcc1,_ =  pt2infty(clustered_ham,ci_vector,pt_vector)
    elcc2,_ =  truncated_pt2(clustered_ham,ci_vector,pt_vector,method = 'enlcc',inf=False)
    assert(abs(elcc1-elcc2)<1e-12)
