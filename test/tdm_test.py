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
np.set_printoptions(suppress=True, precision=10, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

def test_1():
    ttt = time.time()
    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    H      4.00       2.20       2.00
    H      4.10       2.00       0.00
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (0, 0))

    blocks = [[0,1,2,3],[4,5]]
    init_fspace = ((2, 2), (1, 1))
    blocks = [[0,1],[2,3]]
    init_fspace = ((1, 1), (1, 1))
    blocks = [[0,1],[2,3],[4,5]]
    init_fspace = ((1, 1), (1, 1), (1, 1))

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == nelec)
        nelec = cas_nel


    #Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    #cluster using hcore
    #idx = e1_order(h,cut_off = 1e-2)
    #h,g = reorder_integrals(idx,h,g)

    state_i = 0
    state_j = 1

    do_tci = 1

    if do_tci:

        clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, 
                                                            cmf_maxiter     = 20,
                                                            cmf_dm_guess    = None,
                                                            cmf_diis        = False,
                                                            max_roots       = 100,
                                                            delta_elec      = 4
                                                            )

        print(" Build exact eigenstate")
        ci_vector.expand_to_full_space(clusters)
        pt_vector = ci_vector.copy()
        H = build_full_hamiltonian(clustered_ham, ci_vector)
        print(" Diagonalize Hamiltonian Matrix:",flush=True)
        vguess = ci_vector.get_vector()
        if H.shape[0] > 100 and abs(np.sum(vguess)) >0:
            e,v = scipy.sparse.linalg.eigsh(H,n_roots=5,v0=vguess,which='SA')
        else:
            e,v = np.linalg.eigh(H)
        idx = e.argsort()
        e = e[idx]
        e = e + ecore
        v = v[:,idx]

        print(np.linalg.norm(v))
        print(np.linalg.norm(v[:,state_j]))

        ci_vector.zero()
        ci_vector.set_vector(v[:,state_i])
        pt_vector.zero()
        pt_vector.set_vector(v[:,state_j])
        #ci_vector.print_configs()
        pt_vector.print_configs()
        print(pt_vector.norm())
        print(ci_vector.norm())
        rdm_a1, rdm_b1 = build_tdm(ci_vector,pt_vector,clustered_ham)
        print("E i j %4d %4d   %16.8f %16.8f"%(state_i,state_j,e[state_i],e[state_j]))
        print(rdm_a1+rdm_b1)

    do_fci = 1
    if do_fci:
        print("FCI")
        from pyscf import fci
        #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 200 
        cisolver.conv_tol = 1e-14 
        efci, ci = cisolver.kernel(h, g, h.shape[1], nelec=nelec, ecore=ecore,nroots =5,verbose=100)
        #d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
        #print(d1)
        #print("FCIS%10.8f"%(efci))
        #print(t)
        tdm = cisolver.trans_rdm1(ci[state_i], ci[state_j], h.shape[0], nelec=nelec, link_index=None)
        print("E i j %4d %4d   %16.8f %16.8f"%(state_i,state_j,efci[state_i],efci[state_j]))
        print(tdm)
        print("Difference")
        print(tdm - rdm_a1 - rdm_b1)
    
    try:
        assert(np.allclose(rdm_a1 + rdm_b1, tdm, atol=1e-7))
    except:
        assert(np.allclose(rdm_a1 + rdm_b1, -1*tdm, atol=1e-7))
    
if __name__== "__main__":
    test_1() 
