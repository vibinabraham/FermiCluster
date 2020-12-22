import sys, os
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import tools 

from fermicluster import *
from pyscf_helper import *
import pyscf

pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)

def test_1():

    #make hubbard model with 1:1 ratio
    n_orb = 12
    U = 4
    beta1 = 1
    ecore = 0
    h,g = get_hubbard_params(n_orb, beta1, U, pbc=False)
    nelec = n_orb
    C = np.eye(h.shape[0])

    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    init_fspace = ((2, 2), (2, 2),(2, 2))

    #make 3 cluster  
    h[3,4] = -0.25
    h[4,3] = -0.25
    h[7,8] = -0.25
    h[8,7] = -0.25
    print(h)

    # Initialize the CMF solver.
    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100,cs_solver=0) #cs_solver,0 for our FCI and 1 for pyscf FCI solver.
    oocmf.init() # runs a single step CMF calculation
    #oocmf.optimize_orbitals()  # optimize the orbitals using gradient
    oocmf.form_extra_fspace()  #form excited fock space configurations

    clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
    ci_vector = oocmf.ci_vector   # lowest energy TPS using the given Fock space
    h = oocmf.h
    g = oocmf.g
    C = oocmf.C



    ##
    # Excited State TPSCI for the lowest 5 roots. The guess is generated using a CIS type guess, hence there can be other CT states lower which might need different
    #    initialization
    ##

    n_roots = 5

    # STEP 1: Expand the CIS space and generate roots
    ci_vector_s = ci_vector.copy()
    ci_vector_s.add_single_excitonic_states(clustered_ham.clusters)
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector_s)
    e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]

    #STEP 2: Store the first n roots into a clustered_state and prune
    all_vecs = []
    for rn in range(n_roots):
        vec = ci_vector_s.copy()
        vec.zero()
        vec.set_vector(v[:,rn])
        vec.clip(1e-5)
        print("Root:%4d     Energy:%12.8f   Gap:%12.8f  CI Dim: %4i "%(rn,e[rn].real,e[rn].real-e[0].real,len(vec)))
        #vec.print_configs()
        all_vecs.append(vec)

    #STEP 3: Combine all the vecs for each roots into 1 single ci_vector space.
    # Note the coefficients are meanning less here since its multiple states
    for vi,vec in enumerate(all_vecs):
        ci_vector.add(vec)
    ci_vector.zero()
    ci_vector.print()

    #Exta: Print excited state energies after pruning. Just to make sure we have not lost any state
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector, nproc=None)
    e,v = np.linalg.eigh(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    for rn in range(n_roots):
        print("Root:%4d     Energy:%12.8f   Gap:%12.8f"%(rn,e[rn].real,e[rn].real-e[0].real))


    #STEP 4: Run the Excited State-TPSCI and analyze results
    time1 = time.time()
    ci_vector, pt_vector, e0, e2  = ex_tp_cipsi(ci_vector, clustered_ham,  
        thresh_cipsi    = 5e-5, 
        thresh_conv     = 1e-8, 
        max_iter        = 30, 
        n_roots         = n_roots,
        thresh_asci     = 1e-2,
        nbody_limit     = 4, 
        pt_type         = 'en',
        thresh_search   = 1e-6, 
        shared_mem      = 1e8,
        batch_size      = 1,
        matvec          = 3,
        nproc           = None)
    time2 = time.time()

    for rn in range(n_roots):
        print("Root:%4d     Var Energy:%12.8f   Gap:%12.8f  CI Dim: %4i "%(rn,e0[rn].real,e0[rn].real-e0[0].real,len(ci_vector)))
        assert(e0[rn].real-e0[0].real - 0.38556638 )
    for rn in range(n_roots):
        print("Root:%4d     PT  Energy:%12.8f   Gap:%12.8f  CI Dim: %4i "%(rn,e2[rn],e2[rn]-e2[0],len(ci_vector)))

    print("Time spent in the Ex-TPSCI code%16.8f"%(time2-time1))

    var_val = [ 0.00000000,
                0.52992860,
                0.54581756,
                0.55794438,
                1.07016960]

    pt_val = [  0.00000000,
                0.52537536,
                0.53953831,
                0.55320582,
                1.06037191]

    for rn in range(n_roots):
        assert(abs(e0[rn].real-e0[0].real - var_val[rn]) < 1e-7)
        assert(abs(e2[rn].real-e2[0].real - pt_val[rn]) < 1e-7)


if __name__== "__main__":
    test_1() 
