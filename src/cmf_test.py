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
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      2.00       0.00       0.00
    H      3.00       0.00       0.00
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'
    #basis_set = '3-21g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[i] for i in range(4)] 
    init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0))
    #init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))
   
    #blocks = [[0],[1,2],[3]]
    #init_fspace = ((1, 1), (1, 1), (0, 0))
    
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
    n_orb = pmol.n_orb

    h_save = cp.deepcopy(h)
    g_save = cp.deepcopy(g)
    
    print(" Ecore: %12.8f" %ecore)
    
    #cluster using hcore
    #idx = e1_order(h,cut_off = 1e-2)
    #h,g = reorder_integrals(idx,h,g)

    do_fci = 0

    if do_fci:
        #efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore, max_cycle=200, conv_tol=12)
        from pyscf import fci
        #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
        cisolver = fci.direct_spin1.FCI()
        cisolver.max_cycle = 200 
        cisolver.conv_tol = 1e-14 
        efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots=1,verbose=100)
        fci_dim = ci.shape[0]*ci.shape[1]
        d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
        print(" PYSCF 1RDM: ")
        occs = np.linalg.eig(d1)[0]
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        with np.printoptions(precision=6, suppress=True):
            print(d1)
        print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    

    clusters = []
    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))
    
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
   

    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters)
    print(" Add 1-body terms")
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)
    #clustered_ham.combine_common_terms(iprint=1)
   
   
    # Get CMF reference
    cmf(clustered_ham, ci_vector, h, g, max_iter=10)

    dimer_energies = {}
    for i in range(len(blocks)):
        for j in range(i+1,len(blocks)):
            print(" Let's do CMF for blocks %4i:%-4i"%(i,j))
            new_block = []
            new_block.extend(blocks[i])
            new_block.extend(blocks[j])
            new_blocks = [new_block]
            new_init_fspace = [(init_fspace[i][0]+init_fspace[j][0],init_fspace[i][1]+init_fspace[j][1])]
            for k in range(len(blocks)):
                if k!=i and k!=j:
                    new_blocks.append(blocks[k])
                    new_init_fspace.append(init_fspace[k])
            print(" This is the new clustering")
            print(new_blocks)
            new_init_fspace = tuple(new_init_fspace)
            print(new_init_fspace)
            print()
            
            new_clusters = []
            for ci,c in enumerate(new_blocks):
                new_clusters.append(Cluster(ci,c))
            
            new_ci_vector = ClusteredState(new_clusters)
            new_ci_vector.init(new_init_fspace)
           
            
            print(" Clusters:")
            [print(ci) for ci in new_clusters]
            
            new_clustered_ham = ClusteredOperator(new_clusters)
            print(" Add 1-body terms")
            new_clustered_ham.add_1b_terms(cp.deepcopy(h_save))
            print(" Add 2-body terms")
            new_clustered_ham.add_2b_terms(cp.deepcopy(g_save))
            #clustered_ham.combine_common_terms(iprint=1)
           
           
            # Get CMF reference
            e_curr,converged = cmf(new_clustered_ham, new_ci_vector, cp.deepcopy(h_save), cp.deepcopy(g_save), max_iter=10)
            
            print(" Pairwise-CMF(%i,%i) Energy = %12.8f" %(i,j,e_curr))
            dimer_energies[(i,j)] = e_curr

    import operator
    dimer_energies = OrderedDict(sorted(dimer_energies.items(), key=lambda x: x[1]))
    for d in dimer_energies:
        print(" || %10s | %12.8f" %(d,dimer_energies[d]))


if __name__== "__main__":
    test_1() 
