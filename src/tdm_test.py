import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import time
from timeit import default_timer as timer

from fermicluster import *
from pyscf_helper import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=6, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

def test_1():
    ###     PYSCF INPUT
    molecule = '''
    H      0.00       0.00       0.00
    H      1.00       0.00       0.00
    H      2.00       0.00       1.00
    H      3.00       0.00       1.00
    H      4.00       0.00       2.00
    H      5.00       0.00       2.00
    H      6.00       0.00       3.00
    H      7.00       0.00       3.00
    H      8.00       0.00       0.00
    H      9.00       0.00       0.00
    H      10.00      0.00       0.00
    H      11.00      0.00       0.00
    '''
    charge = 0
    spin  = 0
    basis_set = '6-31g'
    basis_set = 'sto-3g'



    blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    blocks = [[0,1],[2,3],[4,5]]
    init_fspace = ((1, 1), (1, 1), (1, 1))
    blocks = [[0,1,2,3],[4,5]]
    init_fspace = ((2, 2), (1, 1))
    blocks = [[0,1],[2,3],[4,5]]
    init_fspace = ((1, 1), (1, 1), (1, 1))
    
    blocks = [[0,1,2,3],[4,5,6,7],[8,9],[10,11]]
    init_fspace = ((2,2),(2,2),(0,1),(1,0))
    
    ###     TPSCI BASIS INPUT
    orb_basis = 'lowdin'
    cas = False
    #cas_nstart = 2
    #cas_nstop = 10
    #cas_nel = 10
    
    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == nelec)
        nelec = cas_nel

    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    

    do_fci = 0
    if do_fci:
        from pyscf import fci
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
    
    clusters, clustered_ham, ci_vector, cmf_out = system_setup(h, g, ecore, blocks, init_fspace, cmf_maxiter = 0, max_roots=8)
    print(clusters[0].basis[(2,2)])
    fock_bra = tuple([(3,2),(1,2),(1,1),(1,1)])
    fock_ket = tuple([(2,2),(2,2),(1,1),(1,1)])
    bra = (0,0,0,0)
    ket = (1,0,0,0)
   
#    print(fock_bra)
#    ci_vector.add_fockspace(fock_bra)
#    
#    delta = ((1,0),(-1,0),(0,0),(0,0))
#    for term in clustered_ham.terms[delta]:
#        print(term)
#
#    print(clustered_ham.terms[delta][0].ints )
#    print(clustered_ham.terms[delta][0].matrix_element(fock_bra, bra, fock_ket, ket) )


    ci_vector.add_fockspace(((3,2),(1,2),(0,1),(1,0)))
    ci_vector.add_fockspace(((3,2),(2,2),(0,1),(0,0)))
    #ci_vector.add_fockspace(((2,3),(2,1),(1,1),(1,1)))
    ci_vector.expand_each_fock_space(clusters)
    

    #for trans in clustered_ham.terms:
    #    for term in clustered_ham.terms[trans]:
    #        if len(term.ints.shape) > 2:
    #            term.ints *= 0
   
    #for c in clusters:
    #    for fock in c.ops["H"]:
    #        c.ops["H"][fock] *= 0

    #edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)

    print(" Build Hamiltonian. Space = ", len(ci_vector), flush=True)
    start = timer()
    H = build_full_hamiltonian_parallel2(clustered_ham, ci_vector)
    stop = timer()
    print(" Time lapse: ",(stop-start))
    n_roots=10
    print(" Diagonalize Hamiltonian Matrix:",flush=True)
    e,v = np.linalg.eig(H)
    idx = e.argsort()
    e = e[idx]
    v = v[:,idx]
    for i,e in enumerate(e[0:min(10,len(e))]):
        print(" %4i %12.8f"%(i+1,e.real))

    #print()
    #print(H)
   
#    fock_l = ((2,2),(2,2),(1,1),(1,1))
#    fock_r = ((3,2),(1,2),(1,1),(1,1))
#    delta_fock= tuple([(fock_l[ci][0]-fock_r[ci][0], fock_l[ci][1]-fock_r[ci][1]) for ci in range(len(clusters))])
#    config_l = (1,0,0,0)
#    config_r = (0,0,0,0)
#    terms = clustered_ham.terms[delta_fock]
#    for term in terms:
#        me = term.matrix_element(fock_l,config_l,fock_r,config_r)
#        print(me)

    #ci_vector.print_configs()

if __name__== "__main__":
    test_1() 
