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

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()




def run(nproc=None):
    pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
    np.set_printoptions(suppress=True, precision=3, linewidth=1500)
    n_cluster_states = 1000

    from pyscf import gto, scf, mcscf, ao2mo

    r0 = 1.5

    molecule= '''
    N       0.00       0.00       0.00
    N       0.00       0.00       {}'''.format(r0)

    charge = 0
    spin  = 0
    basis_set = '6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'boys'
    cas = True
    cas_nstart = 2
    cas_nstop = 10
    cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    #blocks = [[0,1],[2,3],[4,5],[6,7]]
    #init_fspace = ((2, 2), (1, 1), (1, 1), (1, 1))
    blocks = [[0,1,2,3],[4,5],[6,7]]
    init_fspace = ((3, 3), (1, 1), (1, 1))



    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                cas=cas,cas_nstart=cas_nstart,cas_nstop=cas_nstop, cas_nel=cas_nel)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    #cluster using hcore
    idx = e1_order(h,cut_off = 1e-4)
    h,g = reorder_integrals(idx,h,g)


        

    if 0:
        clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, 
                        cmf_maxiter = 20,
                        cmf_diis    = True,
                        cmf_thresh  = 1e-14)


        ci_vector, pt_vector, etci, etci2, conv = bc_cipsi_tucker(ci_vector, clustered_ham, 
                                                            thresh_cipsi        = 1e-6, 
                                                            thresh_ci_clip      = 1e-8, 
                                                            max_tucker_iter     = 0,
                                                            nproc=nproc)
       
        ci_vector.tmp = etci
        file = open('hamiltonian_file_test', 'wb')
        pickle.dump(clustered_ham, file)
        file.close()
        file = open('state_file_test', 'wb')
        pickle.dump(ci_vector, file)
        file.close()

    
    file = open('hamiltonian_file_test', 'rb')
    clustered_ham = pickle.load(file)
    file.close()
    file = open('state_file_test', 'rb')
    ci_vector = pickle.load(file)
    file.close()
    
    compute_pt2_correction(ci_vector, clustered_ham, ci_vector.tmp)
    extrapolate_pt2_correction(ci_vector, clustered_ham, ci_vector.tmp, start=1, stop=1e-3, nsteps=5)
    
    

def test_1():
    run(nproc=None)


if __name__== "__main__":
    test_1() 
