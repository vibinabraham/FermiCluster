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

def test_1():
    molecule = '''
    C  -4.308669   0.197146   0.000000
    C  -1.839087   0.279751   0.000000
    C  -3.110874  -0.411353   0.000000
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

    local = True
    blocks = [range(0,2),range(2,6),range(6,8)] # 3 clusters with 2,4,2 orbitals each
    blocks = [[0,5],[2,3,4,6],[1,7]] # 3 clusters with 2,4,2 orbitals each
    init_fspace = ((1, 1),(2, 2),(1, 1))   # Cluster1: (alpha,beta) Cluster2:(alpha,beta) Cluster3:(alpha,beta)


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



    #h, g = make_stack_lattice(3,4,1,0.6,2,pbc = True)

    #blocks = [range(4),range(4,8),range(8,12)]
    #init_fspace = ((2,2),(2,2),(2,2))
    #nelec = tuple([sum(x) for x in zip(*init_fspace)])

    #C = np.eye(h.shape[0])


    do_fci = 1
    if do_fci:
        # Run a CAS-CI calculation for comparison
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        ecas, vcas = cisolver.kernel(h, g, cas_norb, nelec=cas_nel, ecore=ecore,nroots =1,verbose=100)
        print("CAS-CI:%10.8f"%(ecas))
        print(" CASCI           %12.8f      Dim:%6d" % (ecas,vcas.shape[0]*vcas.shape[1]))
        print(ecore)

    if local:
        ## TPSCI 
        mc = mulliken_ordering(mol,h.shape[0],C)
        idx = np.where(mc>.9)[1]  #gives index map from atom to local orbital corresponding to that orbital

        # Reorder 
        h,g = reorder_integrals(idx,h,g)
        print(h)
        C = C[:,idx] # make sure u reorder this too
        molden.from_mo(mol, 'cas.molden', C)



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

    e_curr, converged, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter = 20)

    # build cluster basis and operator matrices using CMF optimized density matrices
    for ci_idx, ci in enumerate(clusters):
        fspaces_i = ci.possible_fockspaces()

        print()
        print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
        ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=50, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=ecore)

        print(" Build operator matrices for cluster ",ci.idx)
        ci.build_op_matrices(iprint=1)
        ci.build_local_terms(h,g)
        


    opdm_a,opdm_b, tpdm_aa, tpdm_ab, tpdm_ba, tpdm_bb = build_12rdms_cmf(ci_vector,clusters)

    ## Compare energy using density to reference energy

    #compute energy
    opdm = opdm_a + opdm_b
    tpdm = tpdm_aa + tpdm_ab + tpdm_ba + tpdm_bb
    E = np.einsum('pq,pq',h,opdm)
    E += 0.5 * np.einsum('tvuw,tuwv',g,tpdm)

    #reference energy
    e_ref = float(build_hamiltonian_diagonal(clustered_ham,ci_vector))


    ## Compare gradient using matvec to density

    #Generalized Fock
    Gf = np.einsum('pr,rq->pq',h,opdm) + np.einsum('pvuw,quwv->pq',g,tpdm) 
    #Gradient
    Wpq = Gf - Gf.T

    #gradient using matvec (expensive)
    h1_vector = matvec.matvec1(clustered_ham, ci_vector, thresh_search=0, nbody_limit=3)
    rdm_a1, rdm_b1 = build_tdm(ci_vector,h1_vector,clustered_ham)
    rdm_a2, rdm_b2 = build_tdm(h1_vector,ci_vector,clustered_ham)
    Gpq = rdm_a1+rdm_b1-rdm_a2-rdm_b2

    assert(abs(E-e_ref)<1e-8)
    assert(np.allclose(Gpq,Wpq))

if __name__== "__main__":
    test_1() 
