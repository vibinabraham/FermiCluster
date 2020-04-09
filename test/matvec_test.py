import sys, os
sys.path.append('../')
sys.path.append('../src/')
import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
from timeit import default_timer as timer

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()


def test_1():
# {{{
    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000

    from pyscf import gto, scf, mcscf, ao2mo

    r0 = 2.0

    molecule= '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)

    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1,2,7],[3,6],[4,5]]
   
    blocks = [[0,1],[2,3],[4,5,6],[7]]

    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore



    #cluster using hcore
    #idx = e1_order(h,cut_off = 1e-4)
    #h,g = reorder_integrals(idx,h,g)

        
    #efci, fci_dim = run_fci_pyscf(h,g,cas_nel,ecore=ecore)
    #print(" FCI energy: %12.8f" %efci)

        
    init_fspace = ((2,2),(2,2),(2,2),(0,0))
    clusters, clustered_ham, ci_vector = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 0 )

    ci_vector.expand_to_full_space()
    
    np.random.seed(2)
    v = np.random.random([len(ci_vector)]) - .5
    v = v / (v.dot(v))
    ci_vector.set_vector(v)


    print(" Build Hamiltonian. Space = ", len(ci_vector), flush=True)
    H = build_full_hamiltonian_parallel1(clustered_ham, ci_vector)


    out_vector1 = H @ v
    out_vector2 = matvec1(clustered_ham, ci_vector).get_vector()
    out_vector3 = matvec1_parallel1(clustered_ham, ci_vector).get_vector()
    out_vector4 = matvec1_parallel2(clustered_ham, ci_vector).get_vector()
    out_vector5 = matvec1_parallel3(clustered_ham, ci_vector).get_vector()
   
    for vi in range(len(out_vector1)):
        print(" %12.8f %12.8f %12.8f  %12.8f  %12.8f " %(out_vector1[vi],out_vector2[vi],out_vector3[vi],out_vector4[vi],out_vector5[vi]))
        assert(abs(out_vector1[vi]-out_vector2[vi]) < 1e-8)
        assert(abs(out_vector1[vi]-out_vector3[vi]) < 1e-8)
        assert(abs(out_vector1[vi]-out_vector4[vi]) < 1e-8)
        assert(abs(out_vector1[vi]-out_vector5[vi]) < 1e-8)

# }}}



if __name__== "__main__":
    test_1()
