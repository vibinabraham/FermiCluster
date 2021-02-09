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
from pyscf import gto, scf, ao2mo, molden, lo, mo_mapping, mcscf

pyscf.lib.num_threads(1) #with degenerate states and multiple processors there can be issues
np.set_printoptions(suppress=True, precision=3, linewidth=1500)


def form_mol(ndimer,rad,r2):
# {{{
    x = np.zeros(ndimer)
    y = np.zeros(ndimer)

    xa = np.zeros(ndimer)
    ya = np.zeros(ndimer)

    xb = np.zeros(ndimer)
    yb = np.zeros(ndimer)

    angle = 2*np.pi/ndimer


    x[0] = rad
    #print(13)
    #print()
    angle = 2*np.pi/ndimer
    #print("H %8.4f %8.4f %8.4f"%(0,0,0))
    #print("H %8.4f %8.4f %8.4f"%(x[0],y[0],0.000))
    for i in range(1,ndimer):
        x[i] =  rad * np.cos(angle)
        y[i] =  rad * np.sin(angle)
        #print("H %8.4f %8.4f %8.4f"%(x[i],y[i],0.000))
        angle += 2*np.pi/ndimer

    #angle = 2*np.pi/ndimer
    print(2*ndimer+1)
    print()
    angle = 0
    print("Be %8.6f %8.6f %8.6f"%(0,0,0.000))
    molecule = ""
    for i in range(0,ndimer):
        xa[i] = x[i] + .5*r2 * np.cos(angle+(np.pi/2))
        ya[i] = y[i] + .5*r2 * np.sin(angle+(np.pi/2))
        xb[i] = x[i] - .5*r2 * np.cos(angle+(np.pi/2))
        yb[i] = y[i] - .5*r2 * np.sin(angle+(np.pi/2))
        print("H %8.4f %8.4f %8.4f"%(xa[i],ya[i],0.000))
        print("H %8.4f %8.4f %8.4f"%(xb[i],yb[i],0.000))
        angle += 2*np.pi/ndimer
        molecule += "H   {:16.4f}   {:16.4f}   0.0 \n".format(xa[i],ya[i],0)
        molecule += "H   {:16.4f}   {:16.4f}   0.0 \n".format(xb[i],yb[i],0)

    dist1 = np.sqrt((xa[0]-xb[1])**2 + (ya[0]-yb[1])**2) 
    dist = np.sqrt((xa[-1]-xb[0])**2 + (ya[-1]-yb[0])**2) 

    assert(abs(dist - dist1)<1e-12)

    print("Intra distance {:12.4f}         Inter distance {:12.4f}".format(r2,dist))

    return molecule,dist
# }}}


def test_1():

    ndimer = 5
    r2 = 2 #intra bond length
    rad = 3.7  # radius of circle
    molecule,dist = form_mol(ndimer,rad,r2)


    cas_norb = 10
    cas_nel =  10
    nroots = 5

    #PYSCF inputs
    mol = gto.Mole(atom=molecule,
        symmetry = True,basis = 'sto-3g',spin=0)
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
    molden.from_mo(mol, 'cas.molden', C)
    if 0:
        from pyscf import fci
        cisolver = fci.direct_spin1.FCI()
        efci, ci = cisolver.kernel(h, g, h.shape[1], cas_nel, ecore=ecore,nroots =nroots,verbose=100)
        #print(" FCI:        %12.8f"%(27.2114*(efci[0]-efci[1])))
        #print(" FCI:        %12.8f"%(efci))
        for i in range(nroots):
            print(" FCI:     %12.8f   %12.8f"%(efci[i],efci[i]-efci[0]))
        exit()


    mc = mulliken_ordering(mol,h.shape[0],C)
    print(mc.shape)
    idx = np.where(mc>.9)[1]  #gives index map from atom to local orbital corresponding to that orbital
    # Reorder 
    h,g = reorder_integrals(idx,h,g)
    print(h)
    C = C[:,idx] # make sure u reorder this too
    molden.from_mo(mol, 'cas.molden', C)

    blocks = [[0,1],[2,3],[4,5],[6,7],[8,9]] 
    init_fspace = ((1, 1),(1, 1),(1, 1),(1, 1),(1, 1))


    # Initialize the CMF solver. 
    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=10,cs_solver=0) #cs_solver,0 for our FCI and 1 for pyscf FCI solver.
    oocmf.init() # runs a single step CMF calculation
    oocmf.optimize_orbitals()  # optimize the orbitals using gradient
    oocmf.form_extra_fspace(2)  #form excited fock space configurations

    clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
    ci_vector = oocmf.ci_vector   # lowest energy TPS using the reference Fock space
    h = oocmf.h
    g = oocmf.g
    C = oocmf.C



    ##
    # Excited State TPSCI for the lowest 5 roots. The guess is generated using a CIS type guess, hence there can be other CT states lower which might need different
    #    initialization
    ##

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
    for rn in range(nroots):
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
    for rn in range(nroots):
        print("Root:%4d     Energy:%12.8f   Gap:%12.8f"%(rn,e[rn].real,e[rn].real-e[0].real))


    #STEP 4: Run the Excited State-TPSCI and analyze results
    time1 = time.time()
    ci_vector, pt_vector, e0, e2  = ex_tp_cipsi(ci_vector, clustered_ham,  
        thresh_cipsi    = 5e-5, 
        thresh_conv     = 1e-8, 
        max_iter        = 30, 
        n_roots         = nroots,
        thresh_asci     = 1e-2,
        nbody_limit     = 4, 
        pt_type         = 'en',
        thresh_search   = 1e-6, 
        shared_mem      = 1e8,
        batch_size      = 1,
        matvec          = 3,
        nproc           = None)
    time2 = time.time()

    for rn in range(nroots):
        print("Root:%4d     Var Energy:%12.8f   Gap:%12.8f  CI Dim: %4i "%(rn,e0[rn].real,e0[rn].real-e0[0].real,len(ci_vector)))
    for rn in range(nroots):
        print("Root:%4d     PT  Energy:%12.8f   Gap:%12.8f  CI Dim: %4i "%(rn,e2[rn],e2[rn]-e2[0],len(ci_vector)))

    print("Time spent in the Ex-TPSCI code%16.8f"%(time2-time1))

    var_val = [ 0.00000000,
                0.02359112,
                0.02397264,
                0.02397629,
                0.02459324,
                0.02459690]

    pt_val = [  0.00000000,
                0.02329525,
                0.02384732,
                0.02385319,
                0.02482514,
                0.02482831]

    for rn in range(nroots):
        assert(abs(e0[rn].real-e0[0].real - var_val[rn]) < 1e-7)
        assert(abs(e2[rn].real-e2[0].real - pt_val[rn]) < 1e-7)

    assert(len(ci_vector[0]) == 28)

if __name__== "__main__":
    test_1() 
