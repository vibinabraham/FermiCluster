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

def concatenate_cluster_basis(cluster_sets):
    clusters = cp.deepcopy(cluster_sets[0])
    if len(cluster_sets)==1:
        print(" WARNING: Calling concatenate_cluster_basis with a single set, probably not intentional")
        return clusters
    nc = len(clusters)
    #[assert(nc==len(i)) for i in cluster_sets]

    for ci in clusters:
        ci.basis = {}
    ii = 0
    for seti in cluster_sets:
        for ci in range(nc):
            for fi in seti[ci].basis.keys():
                if fi in clusters[ci].basis.keys():
                    #print(ci, fi, ii)
                    
                    #print(" Nick1: ", clusters[ci].basis[fi].shape) 
                    #print(" Nick1: ", seti[ci].basis[fi].shape, clusters[ci].basis[fi].shape) 
                    clusters[ci].basis[fi] = np.hstack((clusters[ci].basis[fi], seti[ci].basis[fi]))
                    #print(" Nick2: ", clusters[ci].basis[fi].shape) 
                else:
                    clusters[ci].basis[fi] = 1*seti[ci].basis[fi]
                #print(" Nick2: ", clusters[ci].basis[fi].shape) 
        ii+=1
    for ci in clusters:
        ci.ovlp = {}
        print("\n")
        print(ci)
        for fi in ci.basis:
            print("basis:", fi)
            print(ci.basis[fi].shape)
            ovlp = ci.basis[fi].T @ ci.basis[fi]
            l,U = np.linalg.eigh(ovlp)
            idx = l.argsort()[::-1]
            l = l[idx]
            U = U[:,idx]
            nkeep = 0
            for li in l:
                print(" nick:i",li)
                if abs(li) > 1e-8:
                    U[:,nkeep] /= np.sqrt(li)
                    nkeep += 1
            ci.basis[fi] = ci.basis[fi] @ U[:,:nkeep]
            ci.ovlp[fi] = ci.basis[fi].T @ ci.basis[fi]

            print(ci.ovlp[fi])
    return clusters 

def test_truncate_basis():
    # {{{
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
    H      0.70       0.00       0.00
    H      0.00       1.00       0.00
    H      0.70       1.00       0.00
    H      0.00       2.00       0.00
    H      0.70       2.00       0.00
    H      0.00       3.00       0.00
    H      0.70       3.00       0.00
    '''
    charge = 0
    spin  = 0
    basis_set = '3-21g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    init_fspace = ((1, 1), (1, 1), (1, 1))
    blocks = [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]
    init_fspace = ((2, 2), (2, 2))
    fspaces = []
    fspaces.append(init_fspace)
    
    if 1:
        for bi in range(len(blocks)):
            for bj in range(len(blocks)):
                if bi == bj:
                    continue
        
                if init_fspace[bi][0] > 0 and init_fspace[bj][0] < len(blocks[bj]): 
                    new_fock = [[init_fspace[i][0], init_fspace[i][1]] for i in range(len(blocks))]
                    new_fock[bi][0] -= 1
                    new_fock[bj][0] += 1
                    new_fock = tuple([tuple(i) for i in new_fock])
                    fspaces.append(new_fock)
        
        
                if init_fspace[bi][1] > 0 and init_fspace[bj][1] < len(blocks[bj]): 
                    new_fock = [[init_fspace[i][0], init_fspace[i][1]] for i in range(len(blocks))]
                    new_fock[bi][1] -= 1
                    new_fock[bj][1] += 1
                    new_fock = tuple([tuple(i) for i in new_fock])
                    fspaces.append(new_fock)
                
                    if init_fspace[bi][0] > 0 and init_fspace[bj][0] < len(blocks[bj]): 
                        new_fock = [[init_fspace[i][0], init_fspace[i][1]] for i in range(len(blocks))]
                        new_fock[bi][0] -= 1
                        new_fock[bi][1] -= 1
                        new_fock[bj][0] += 1
                        new_fock[bj][1] += 1
                        new_fock = tuple([tuple(i) for i in new_fock])
                        fspaces.append(new_fock)
                



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

    print(" Ecore: %12.8f" %ecore)

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
    

    
    n_blocks = len(blocks)
    clusters = [Cluster(ci,c) for ci,c in enumerate(blocks)]
    print(" Clusters:")
    [print(ci) for ci in clusters]
    
    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    clustered_ham.h = h
    clustered_ham.g = g


    #fspaces = [init_fspace1, init_fspace2, init_fspace3, init_fspace4, init_fspace5, init_fspace6, init_fspace7]
    #fspaces = [init_fspace1, init_fspace2, init_fspace3, init_fspace4, init_fspace5]
    cluster_sets = []

    for fspace in fspaces:
        ci_vector = ClusteredState()
        ci_vector.init(clusters, fspace)
        
        max_roots = 10 
        e_cmf, cmf_conv, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, 
                diis        = 0,
                diis_start  = 1,
                thresh      = 1e-9,
                max_iter    = 20)
        # build cluster basis and operator matrices using CMF optimized density matrices
        for ci_idx, ci in enumerate(clusters):
            fspaces_i = fspace[ci_idx]
            fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], 0) )
            print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
            ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=max_roots, rdm1_a=rdm_a, rdm1_b=rdm_b, ecore=ecore)
            #print(" Build operator matrices for cluster ",ci.idx)
            #ci.build_op_matrices()
            #ci.build_local_terms(h,g)
            print(ci)
            for f in ci.basis:
                print(f)
                print(ci.basis[f])
        cluster_sets.append(cp.deepcopy(clusters))
 

    clusters = concatenate_cluster_basis(cluster_sets)
        
    ci_vector = ClusteredState()
    ci_vector.init(clusters, init_fspace)
    
    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    clustered_ham.h = h
    clustered_ham.g = g
    
    # build cluster basis and operator matrices using CMF optimized density matrices
    for ci_idx, ci in enumerate(clusters):
        print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)

    [print(i) for i in fspaces]
    ci_vector.expand_to_full_space(clusters)
    ci_vector.print()
    print(len(ci_vector))
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)
    e,v = np.linalg.eigh(H)

    e = e[0]
    print(e)
    assert(np.isclose(e,-4.289958228357164,atol=1e-12))
    # }}}


if __name__== "__main__":
    test_truncate_basis() 
