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
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

def test_1():
    r0 = 2.00 
    molecule = '''
    H   0   0   0
    H   0   0   1
    H   0   0   2
    H   0   0   3
    H   0   0   6
    H   0   0   7
    H   0   0   8
    H   0   0   9
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    init_fspace = ((2, 2), (2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,cas=False)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    do_fci = 0
    do_hci = 0
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

    #cluster using hcore
    #idx = e1_order(h,cut_off = 2e-1)
    #h,g = reorder_integrals(idx,h,g)
    if do_tci:
        n_blocks = len(blocks)

        clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 20 )


        #ci_vector.expand_to_full_space()
        #ci_vector.expand_each_fock_space()
        #ci_vector.add_single_excitonic_states()
        #ci_vector.print_configs()

        precompute_cluster_basis_energies(clustered_ham)
        edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
        print(edps)
        emp2,pt_vector = compute_rspt2_correction(ci_vector, clustered_ham, edps, nproc=1)
        #elcc,pt_vector = compute_lcc2_correction(ci_vector, clustered_ham, edps, nproc=1)
        een2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, edps, nproc=1)
        ecepa = cepa(clustered_ham,ci_vector,pt_vector,cepa_shift='cepa0')
        ecisd = cepa(clustered_ham,ci_vector,pt_vector,cepa_shift='cisd')
        ecisd2 = compute_cisd_correction(ci_vector, clustered_ham, nproc=1)
        print(" DPS        :       %12.8f      Dim:%6d" % (edps,1))
        print(" DPS-MPPT2  :       %12.8f      Dim:%6d" % (edps+emp2,len(pt_vector)))
        print(" DPS-ENPT2  :       %12.8f      Dim:%6d" % (edps+een2,len(pt_vector)))
        print(" DPS-CEPA0  :       %12.8f      Dim:%6d" % (edps+ecepa,len(pt_vector)))
        print(" DPS-CISD   :       %12.8f      Dim:%6d" % (edps+ecisd,len(pt_vector)))
        print(" DPS-CISD2  :       %12.8f      Dim:%6d" % (edps+ecisd2,len(pt_vector)))
        print(" DPS        :       %12.8f      Dim:%6d" % (edps+ecore,1))
        print(" DPS-MPPT2  :       %12.8f      Dim:%6d" % (edps+emp2+ecore,len(pt_vector)))
        print(" DPS-ENPT2  :       %12.8f      Dim:%6d" % (edps+een2+ecore,len(pt_vector)))
        print(" DPS-CEPA0  :       %12.8f      Dim:%6d" % (edps+ecepa+ecore,len(pt_vector)))
        print(" DPS-CISD   :       %12.8f      Dim:%6d" % (edps+ecisd+ecore,len(pt_vector)))
        print(" DPS-CISD2  :       %12.8f      Dim:%6d" % (edps+ecisd2+ecore,len(pt_vector)))
        tci_dim = len(pt_vector)
        #pt_vector.clip(.005)
        pt_vector.print_configs()
        edps = edps+ecore
        emp2 = edps+emp2
        een2 = edps+een2
        ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,selection='cipsi',
            thresh_cipsi=1e-8, thresh_ci_clip=1e-8, max_tucker_iter=4)
        print(" TCI     :       %12.8f      Dim:%6d" % (etci+ecore,1))
        print(" TCI     :       %12.8f      Dim:%6d" % (etci2+ecore,1))
        print(" DPS-MPPT2  :       %12.8f      Dim:%6d" % (emp2,len(pt_vector)))
        print(" DPS-ENPT2  :       %12.8f      Dim:%6d" % (een2,len(pt_vector)))
        print(" DPS-CEPA0  :       %12.8f      Dim:%6d" % (edps+ecepa,len(pt_vector)))
        print(" DPS-CISD   :       %12.8f      Dim:%6d" % (edps+ecisd,len(pt_vector)))
        print(" DPS-CISD2  :       %12.8f      Dim:%6d" % (edps+ecisd2,len(pt_vector)))
        assert(np.abs(ecisd - ecisd2) <1e-8)
        assert(np.abs(emp2  --4.25536374 ) <1e-8)
        assert(np.abs(een2  --4.30347123 ) <1e-8)
        assert(np.abs(ecepa+edps --4.44499102 ) <1e-8)
        assert(np.abs(ecisd+edps --4.31115873 ) <1e-8)

if __name__== "__main__":
    test_1() 
