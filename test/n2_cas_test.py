import sys, os
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *
from bc_cipsi import *
import pyscf
ttt = time.time()


def test_1():
    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000

    from pyscf import gto, scf, mcscf, ao2mo

    r0 = 1.5

    mol = gto.Mole()
    mol.atom = '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)

    mol.charge = +0
    mol.spin = +0
    mol.max_memory = 1000  # MB

    mol.basis = '6-31g'

    #mol.symmetry =True

    myhf = scf.RHF(mol).run()
    print(myhf.mo_energy)
    #exit()
    n_orb = myhf.mo_coeff.shape[1]
    #g.shape = (n_orb,n_orb,n_orb,n_orb)
    enu = myhf.energy_nuc()
    S = mol.intor('int1e_ovlp_sph')

    local = 'boys'
    #local = 'p'



    def e1_order(h):
    # {{{
        hnew = np.absolute(h)
        hnew[hnew < 1e-4] = 0
        np.fill_diagonal(hnew, 0)
        print(hnew)
        import scipy.sparse
        idx = scipy.sparse.csgraph.reverse_cuthill_mckee(
            scipy.sparse.csr_matrix(hnew))
        print(idx)
        idx = idx 
        hnew = h[:, idx]
        hnew = hnew[idx, :]
        print(hnew)
        return idx
    # }}}

    if local == 'boys':
        from pyscf import lo, molden

        n_start =  2
        n_stop  = 10
        nelec   = 10
        norb = n_stop - n_start

        print("\nUsing localised orbitals...\n")
        cl_c = myhf.mo_coeff[:, :n_start]
        cl_a = lo.Boys(mol, myhf.mo_coeff[:, n_start:n_stop]).kernel(verbose=4)
        cl_v = myhf.mo_coeff[:, n_stop:]

        Cl = np.column_stack((cl_c, cl_a, cl_v))

        molden.from_mo(mol, 'h8.molden', Cl)
        h = Cl.T.dot(myhf.get_hcore()).dot(Cl)
        g = ao2mo.kernel(mol, Cl, aosym='s4',compact=False).reshape(4 * ((n_orb), ))

        print(myhf.mo_coeff)
        print(Cl)

        print(cl_a.shape)

        mycas = mcscf.CASSCF(myhf, norb, nelec)
        h1e_cas, ecore = mycas.get_h1eff(mo_coeff = Cl)   #needs the core orbs also to form the ecore and eff
        h2e_cas = mycas.get_h2eff(mo_coeff = Cl)
        h2e_cas = mycas.get_h2cas(mo_coeff = Cl)
        h2e_cas = ao2mo.kernel(mol, cl_a, aosym='s4',compact=False).reshape(4 * ((norb), ))
        print(h1e_cas)

        
        idx = e1_order(h1e_cas)
        h1e_cas = h1e_cas[:,idx] 
        h1e_cas = h1e_cas[idx,:] 
        #print(h1e_cas.shape)
        #print(h2e_cas.shape)

        h2e_cas = h2e_cas[:,:,:,idx] 
        h2e_cas = h2e_cas[:,:,idx,:] 
        h2e_cas = h2e_cas[:,idx,:,:] 
        h2e_cas = h2e_cas[idx,:,:,:] 


    do_fci = 1
    do_hci = 1


    if do_fci:
        # FCI
        from pyscf import fci
        efci, ci = fci.direct_spin1.kernel(h1e_cas, h2e_cas, norb, nelec,ecore=ecore, verbose=5)
        fci_dim = ci.shape[0] * ci.shape[1]
        print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
        print("FCI %10.8f" % (efci + enu))

        mc = mcscf.CASCI(myhf, 8, 10)
        mc.kernel(verbose=100)
        ecas = mc.e_tot 
        casdim = mc.ci.shape[0] * mc.ci.shape[1]
        print(casdim)

    if do_hci:
        #heat bath ci
        from pyscf import mcscf
        from pyscf.hci import hci
        cisolver = hci.SCI(mol)
        cisolver.select_cutoff = 1e-3
        cisolver.ci_coeff_cutoff = 1e-3
        ehci, civec = cisolver.kernel(h1e_cas,h2e_cas,h1e_cas.shape[1],nelec,ecore=ecore,verbose=100)
        hci_dim = civec[0].shape[0]
        print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
        print("HCI %10.8f" % (ehci + enu))


    #blocks = [[0,1,2,3],[4,5,6,7]]
    #blocks = [[0,1],[2,3],[4,5,6],[7,8,9],range(10,14),range(14,18),range(18,22),range(22,25),range(25,28)]
    ##blocks = [[0,1],[2,3],[4,5,6],[7,8,9]]
    blocks = [[0,1,2,3],[4,5],[6,7]]
    n_blocks = len(blocks)
    clusters = []

    for ci, c in enumerate(blocks):
        clusters.append(Cluster(ci, c))

    ci_vector = ClusteredState(clusters)
    #ci_vector.init(((2,2),(0,0)))
    ci_vector.init(((3, 3), (1, 1), (1, 1)))

    if mol.basis == 'sto-3g':
        #ci_vector.init(((2,2),(3,2),(2,3)))
        #ci_vector.init(((2,2),(2,3),(3,2)))
        ci_vector.init(((2, 2), (3, 3), (1, 1), (1, 1)))
        #ci_vector.init(((2,2),(2,2),(1,1),(2,2)))
        #ci_vector.init(((2,2),(2,2),(3,1),(2,0)))
        #ci_vector.init(((2,2),(2,2),(1,3),(0,2)))
        #ci_vector.init(((2,2),(2,2),(1,2),(2,1)))
        #ci_vector.init(((2,2),(2,2),(2,1),(1,2)))

    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h1e_cas)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(h2e_cas)
    #clustered_ham.combine_common_terms(iprint=1)

    print(" Build cluster basis")
    for ci_idx, ci in enumerate(clusters):
        assert (ci_idx == ci.idx)
        print(" Extract local operator for cluster", ci.idx)
        opi = clustered_ham.extract_local_operator(ci_idx)
        print()
        print()
        print(" Form basis by diagonalize local Hamiltonian for cluster: ", ci_idx)
        ci.form_eigbasis_from_local_operator(opi, max_roots=n_cluster_states)

    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ", c.idx)
        c.build_op_matrices()

    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()

    #ci_vector, pt_vector, e0, e2 = bc_cipsi_tucker(ci_vector.copy(),clustered_ham)
    ci_vector, pt_vector, e0, e2 = bc_cipsi_tucker(ci_vector.copy(),clustered_ham,hshift=1e-8, max_tucker_iter=20, thresh_cipsi=1e-5, thresh_ci_clip=1e-6)
    #ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector.copy(),clustered_ham, thresh_cipsi=1e-5, thresh_ci_clip=1e-5)
    bcci_dim = len(ci_vector)

    print("  radius:        %12.8f " % r0)
    print("    E Nu:        %12.8f " % enu)
    print("    BCCI:        %12.8f Dim:%6d" % (e0 + ecore, bcci_dim))
    print("    BCCI(2):     %12.8f Dim:%6d" % (e2 + ecore, bcci_dim))
    print("     HCI:        %12.8f Dim:%6d" % (ehci , hci_dim))
    print("     FCI:        %12.8f Dim:%6d" % (efci , fci_dim))
    print("CAS(6,6):        %12.8f Dim:%6d" % (ecas , casdim))
    assert(abs(e0 + ecore --108.85558051)< 1e-7)
    assert(abs(e2 + ecore --108.85574547)< 1e-7)
    assert(abs(bcci_dim - 67)<1e-15)
    assert(abs(efci   --108.85574521)< 1e-7)


if __name__== "__main__":
    test_1() 
