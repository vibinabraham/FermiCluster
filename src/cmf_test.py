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
    # {{{
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      2.00       0.00       0.00
    H      3.00       0.00       0.00
    '''
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    molecule = '''
    N      0.00       0.00       0.00
    N      0.00       0.00       2.00
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'
    basis_set = '6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 18
    cas_nel = 10
    loc_nstart = 2
    loc_nstop = 18
    

    ###     TPSCI CLUSTER INPUT
    #blocks = [[i] for i in range(4)] 
    #init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0))
    #blocks = [[i] for i in range(8)] 
    #init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))
    blocks = [[i] for i in range(16)] 
    occ_fspace = [(1,1) for i in range(5)]
    vir_fspace = [(0,0) for i in range(11)]
    occ_fspace.extend(vir_fspace)
    init_fspace = tuple( occ_fspace) 
    


    #Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,cas,cas_nstart,cas_nstop,cas_nel,loc_nstart,loc_nstop)

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
    e0, converged = cmf(clustered_ham, ci_vector, h, g, max_iter=10)
    e2, v = compute_pt2_correction(ci_vector, clustered_ham, e0)
    print(" PT2 Energy Total      = %12.8f" %(e0+e2))
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=False)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=False)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=False)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=False)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=False)
    
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    blocks, init_fspace = do_2body_search(blocks, init_fspace, h, g, max_cluster_size=4, do_pt2=True)
    return
   # }}}


def test_2():
    # {{{
    ttt = time.time()

    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      2.00       0.00       0.00
    H      3.00       0.00       0.00
    '''
    molecule = '''
    H      0.00       0.00       0.00
    H      2.00       0.00       2.00
    H      0.00       2.20       2.00
    H      2.10       2.00       0.00
    '''
    molecule = '''
    N      0.00       0.00       0.00
    N      0.00       0.00       2.00
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'
    basis_set = '6-31g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop = 18
    cas_nel = 10
    loc_nstart = 2
    loc_nstop = 18

    ###     TPSCI CLUSTER INPUT
    #blocks = [[i] for i in range(4)] 
    #init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0))
    blocks = [[i] for i in range(8)] 
    init_fspace = ((1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))
    
    blocks = [[0,2,4,6],[1,3,5,7]] 
    init_fspace = ((1, 1), (1, 1))
    
    blocks = [[0,1,2,3],[4,5,6,7]] 
    init_fspace = ((2, 2), (0, 0))
    
    blocks = [[0,1,4,5],[2,3,6,7]] 
    init_fspace = ((2, 2), (0, 0))
    
    blocks = [[0,1,2,3],[4],[5],[6],[7]] 
    init_fspace = ((2, 2), (0, 0), (0, 0), (0, 0), (0, 0))

    blocks = [[0, 4, 1, 5], [2, 7, 3, 6]]
    init_fspace = ((2, 2), (2, 2))
    
    blocks = [[0, 1, 2, 7], [3, 4, 5, 6]]
    init_fspace = ((2, 2), (2, 2))
    
    blocks =  [[0, 12, 8, 11], [2, 7, 10, 14], [1, 9, 13, 15], [4, 5, 3, 6]]
    init_fspace = ((1, 1), (1, 1), (1, 1), (2, 2))
    
    blocks =  [[0,1,11,15],[2,7,10,14],[3,6,8,13],[4,5,9,12]]
    init_fspace = ((2, 2), (1,1), (1,1), (1,1))
    
    #Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,cas,cas_nstart,cas_nstop,cas_nel,loc_nstart,loc_nstop)

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
        d2 = cisolver.make_rdm2(ci, h.shape[1], nelec)
        print(" PYSCF 1RDM: ")
        occs = np.linalg.eig(d1)[0]
        [print("%4i %12.8f"%(i,occs[i])) for i in range(len(occs))]
        with np.printoptions(precision=6, suppress=True):
            print(d1)
        print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    
        norb = h.shape[1]
        #cumu = np.kron(d1,d1)
        #cumu.shape = (norb,norb,norb,norb)
        #d2 = d2 - .25*cumu
        d2 = g
        A = 0*d1
        for p in range(norb):
            for q in range(norb):
                for r in range(norb):
                    for s in range(norb):
                        A[p,q] += abs(d2[p,q,r,s])
                        A[p,r] += abs(d2[p,q,r,s])
                        A[p,s] += abs(d2[p,q,r,s])
                        A[q,r] += abs(d2[p,q,r,s])
                        A[q,s] += abs(d2[p,q,r,s])
                        A[r,s] += abs(d2[p,q,r,s])
                        A[q,p] += abs(d2[p,q,r,s])
                        A[r,p] += abs(d2[p,q,r,s])
                        A[s,p] += abs(d2[p,q,r,s])
                        A[r,q] += abs(d2[p,q,r,s])
                        A[s,q] += abs(d2[p,q,r,s])
                        A[s,r] += abs(d2[p,q,r,s])

        A = abs(np.einsum('pqrs,rs->pq',g,d1))
        A = abs(h) 
        A = abs(np.einsum('pqrs,qr->ps',g,d1))
        A = A / np.max(np.max(A))
        print_mat(A)
        
        import sklearn.cluster as cluster
        from sklearn.cluster import SpectralClustering
        import matplotlib.pyplot as plt 
        if 0:
            plt.matshow(A);
            plt.colorbar()
            plt.show()
        
        D = sum(A)
        L = np.diag(D) - A
        Li,Lv = np.linalg.eigh(L)
        idx = Li.argsort()
        Lv = Lv[:,idx]
        Li = Li[idx]

        for ei in range(Li.shape[0]):
            if ei==0:
                print(" %4i Eval = %12.8f" %(ei+1,Li[ei]))
            else:
                print(" %4i Eval = %12.8f Gap = %12.8f" %(ei+1,Li[ei],Li[ei]-Li[ei-1]))
         
        print(" Fiedler vector")
        fv = Lv[:,1]
        idx = fv.argsort()
        labels = np.array(range(Lv.shape[0]))[idx]
        fv = fv[idx]
        for i in range(fv.shape[0]):
            print(" %4i %4i %12.8f" %(i,labels[i],fv[i]))

        print(" Bisection:")
        
        print(" Now do k-means clustering")
        kmeans = cluster.KMeans(n_clusters=2)
        kmeans.fit(A)
        print(kmeans.labels_)
        
        print(" Now do spectral clustering")
        clustering = SpectralClustering(n_clusters=2,affinity='precomputed').fit(A)
        #clustering = SpectralClustering(n_clusters=2,random_state=0, affinity='precomputed').fit(A)
        print(clustering.labels_)
        import networkx as nx
        def draw_graph(G):
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            plt.show()
        
        G=nx.from_numpy_matrix(A)
        draw_graph(G)
        exit() 

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
  
    nproc = 1
    
    # Get CMF reference
    e0, cmf_converged = cmf(clustered_ham, ci_vector, h, g, max_iter=10)

    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector, clustered_ham,
        thresh_ci_clip=1e-7,thresh_cipsi=1e-3,hshift=1e-8,max_tucker_iter=20)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector, clustered_ham,
        thresh_ci_clip=1e-7,thresh_cipsi=1e-4,hshift=1e-8,max_tucker_iter=20)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector, clustered_ham,
        thresh_ci_clip=1e-7,thresh_cipsi=1e-5,hshift=1e-8,max_tucker_iter=20)
    exit()


    if cmf_converged == False:
        print(" CMF didn't converged")
        exit()

    print(" Compute Matrix Vector Product:", flush=True)
    start = time.time()
    if nproc==1:
        pt_vector = matvec1(clustered_ham, ci_vector)
    else:
        pt_vector = matvec1_parallel1(clustered_ham, ci_vector, nproc=nproc)
    stop = time.time()
    print(" Time spent in matvec: ", stop-start)
    
    pt_vector.prune_empty_fock_spaces()

    
    tmp = ci_vector.dot(pt_vector)
    var = pt_vector.norm() - tmp*tmp 
    print(" Variance: %12.8f" % var,flush=True)


    print(" Remove CI space from pt_vector vector")
    for fockspace,configs in pt_vector.items():
        if fockspace in ci_vector.fblocks():
            for config,coeff in list(configs.items()):
                if config in ci_vector[fockspace]:
                    del pt_vector[fockspace][config]


    for fockspace,configs in ci_vector.items():
        if fockspace in pt_vector:
            for config,coeff in configs.items():
                assert(config not in pt_vector[fockspace])

    print(" Norm of CI vector = %12.8f" %ci_vector.norm())
    print(" Dimension of CI space: ", len(ci_vector))
    print(" Dimension of PT space: ", len(pt_vector))
    print(" Compute Denominator",flush=True)
    #next_ci_vector = cp.deepcopy(ci_vector)
    # compute diagonal for PT2

    start = time.time()
    pt_vector.prune_empty_fock_spaces()
        
    #import cProfile
    #pr = cProfile.Profile()
    #pr.enable()
       
    if nproc==1:
        Hd = build_hamiltonian_diagonal(clustered_ham, pt_vector)
    else:
        Hd = build_hamiltonian_diagonal_parallel1(clustered_ham, pt_vector, nproc=nproc)
    #pr.disable()
    #pr.print_stats(sort='time')
    end = time.time()
    print(" Time spent in demonimator: ", end - start)

    denom = 1/(e0 - Hd)
    pt_vector_v = pt_vector.get_vector()
    pt_vector_v.shape = (pt_vector_v.shape[0])

    e2 = np.multiply(denom,pt_vector_v)
    pt_vector.set_vector(e2)
    e2 = np.dot(pt_vector_v,e2)

    print(" PT2 Energy Correction       = %12.8f" %(e2))
    print(" TPSCI+PT2 Energy Total      = %12.8f" %(e0+e2+ecore))
   

    # Now add pt_vector to ci_vector, normalize and get 1rdm
    ci_vector.add(pt_vector)
    ci_vector.normalize()
    rdm_a, rdm_b = tools.build_1rdm(ci_vector)
   
    print_mat(rdm_a+rdm_b)


    import sklearn.cluster as cluster
    from sklearn.cluster import SpectralClustering
    import matplotlib.pyplot as plt 
    A = abs(rdm_a + rdm_b)
    D = sum(A)
    L = np.diag(D) - A
    Li,Lv = np.linalg.eigh(L)
    idx = Li.argsort()
    Lv = Lv[:,idx]
    Li = Li[idx]
    for ei in range(Li.shape[0]):
        if ei==0:
            print(" %4i Eval = %12.8f" %(ei+1,Li[ei]))
        else:
            print(" %4i Eval = %12.8f Gap = %12.8f" %(ei+1,Li[ei],Li[ei]-Li[ei-1]))
    
    print(" Fiedler vector")
    for i in range(Li.shape[0]):
        print(" %4i %12.8f" %(i,Lv[i,1]))
    
    print(" Now do k-means clustering")
    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(A)
    print(kmeans.labels_)
    
    print(" Now do spectral clustering")
    clustering = SpectralClustering(n_clusters=2,affinity='precomputed').fit(A)
    #clustering = SpectralClustering(n_clusters=2,random_state=0, affinity='precomputed').fit(A)
    print(clustering.labels_)

    return
   # }}}

if __name__== "__main__":
    #test_1() 
    test_2() 
