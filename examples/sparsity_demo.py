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
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

np.set_printoptions(suppress=True, precision=3, linewidth=1500)

dim_a = 2 #n cluster
dim_b = 4 # site per cluster
U = 5
beta1 = 1
beta2 = 1/10
print(beta2)
n_orb = dim_a * dim_b

h, g = make_stack_lattice(dim_a,dim_b,beta1,beta2,U,pbc = True)
np.save('h_local.npy',h)

blocks = [range(4),range(4,8)]
init_fspace = ((2,2),(2,2))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
out_str = "H_cmf.npy"
    

deloc = True 
deloc = False 
if deloc:
	# get scf orbitals 
	Escf,orb,h,g,C = run_hubbard_scf(h,g,n_orb//2)
	blocks = [range(i,i+1) for i in range(8)] 
	init_fspace = ((1,1),(1,1),(1,1),(1,1),(0,0),(0,0),(0,0),(0,0))
	nelec = tuple([sum(x) for x in zip(*init_fspace)])
	out_str = "H_hf.npy"
	np.save('h_scf.npy',h)

do_fci = 0
do_hci = 0
do_tci = 1

if do_fci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    Escf,orb,h2,g2,C = run_hubbard_scf(h,g,n_orb//2)
    efci, fci_dim = run_fci_pyscf(h2,g2,nelec,ecore=0)
if do_hci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    Escf,orb,h2,g2,C = run_hubbard_scf(h,g,n_orb//2)
    ehci, hci_dim = run_hci_pyscf(h2,g2,nelec,ecore=0,select_cutoff=5e-4,ci_cutoff=5e-4)
if do_tci:
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

    ci_vector.expand_to_full_space()
    H = build_full_hamiltonian(clustered_ham, ci_vector)
    np.save(out_str,H)
    
    exit() 
    ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=0,
        thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=10,max_cipsi_iter=20)
    ci_vector.print_configs()
    tci_dim = len(ci_vector)


if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (etci, tci_dim))

