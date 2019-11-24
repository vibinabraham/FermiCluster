import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys
import time
from hubbard_fn import *

from Cluster import *
from ClusteredOperator import *
from ClusteredState import *
from tools import *
from bc_cipsi import *

def run_tpsci(h, g, blocks, init_fspace, 
                    ecore=0, 
                    thresh_cipsi=1e-4, 
                    thresh_ci_clip=1e-5, 
                    thresh_cipsi_conv=1e-8,
                    max_cipsi_iter=30, 
                    thresh_tucker_conv= 1e-6, 
                    max_tucker_iter=20, 
                    tucker_state_clip=None,
                    cs_guess = None,
                    hshift=1e-8):
    """
    Tensor Product Selected Configuration Interaction (TPSCI)

    Parameters
    -----------
    h : 2d numpy array 
        One electron integral     
    g  : 4d numpy array
        Two electron integral (chemist notation) 
    blocks : list of list 
        Orbital block definition for the given system 
        eg. [[0,2],[1,3]] two blocks with orbitals 0,2 in a cluster, and orbitals 1,3 in another
    init_fspace : tuple of tuples 
        Defining the initial fock space for the cluster
        eg. ((2,1),(0,1)) would initialize the 2 alpha 1 beta in cluster A and 0 alpha 1 beta in cluster B.
    ecore : float
        Nuclear repulsion + frozen core number in case of a cas calculation
    thresh_cipsi : float
        the thresh for coeff^2 in the first order PT space
    thresh_ci_clip : float
        If the coefficient is below this value for a determinant, it gets clipped 
    thresh_cipsi_conv : float
        Convergence threshold for cipsi iteration
    thresh_tucker_conv : float
        Convergence threshold for tucker iterations
    tucker_state_clip : float
        Clip the tucker state, improve the tucker iterations_
    hshift : float
        For the tucker iterations, shift the BRDM rotation by hshift* H for the cluster in the fock space


    Examples
    -----------
        For He dimer in 6-31g basis
        we have 4 orbitals and 4 electrons
            Using the lowdin orthogonal orbitals: 
                defining 1s and 2s in atom 1 as a block and 1s and 2s of atom 2 as another.
                blocks = [[0,1],[2,3]]
                init_fspace = ((1,1),(1,1))
            Using RHF orbitals: 
                defining the occupied orbitals as a cluster and virtual as another
                blocks = [[0,1],[2,3]]
                init_fspace = ((2,2),(0,0))

    """
# {{{
    print("      ==================================================================== ")
    print("     |                                                                    |")
    print("     |                                                                    |")
    print("     |         Tensor Product Selected Configuration Interaction          |")
    print("     |                                                                    |")
    print("     |                                                                    |")
    print("      ==================================================================== ")
    print("         Options:")
    for k,v in dict(blocks=blocks, init_fspace= init_fspace, 
                ecore=ecore, 
                thresh_cipsi=thresh_cipsi, 
                thresh_ci_clip=thresh_ci_clip, 
                thresh_cipsi_conv=thresh_cipsi_conv,
                max_cipsi_iter=max_cipsi_iter, 
                thresh_tucker_conv= thresh_tucker_conv,
                max_tucker_iter=max_tucker_iter, 
                tucker_state_clip=tucker_state_clip).items():
        if v == None:
            v = 'None'
        #print("{:<25}{:.>100}".format(k,v))
        print('                 {:.<25}      {}'.format(k, v))

    n_blocks = len(blocks)

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

    print(" Build cluster basis")
    for ci_idx, ci in enumerate(clusters):
        assert(ci_idx == ci.idx)
        print(" Extract local operator for cluster",ci.idx)
        opi = clustered_ham.extract_local_operator(ci_idx)
        print()
        print()
        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
        ci.form_eigbasis_from_local_operator(opi,max_roots=1000)


    #clustered_ham.add_ops_to_clusters()
    print(" Build these local operators")
    for c in clusters:
        print(" Build mats for cluster ",c.idx)
        c.build_op_matrices()

    #ci_vector.expand_to_full_space()
    #ci_vector.expand_each_fock_space()
    #ci_vector.add_single_excitonic_states()
    #ci_vector.print_configs()

    if max_tucker_iter ==0:
        ci_vector, pt_vector, e0, e2 = bc_cipsi(ci_vector.copy(), clustered_ham, 
                thresh_cipsi=thresh_cipsi, thresh_ci_clip=thresh_ci_clip, thresh_conv=thresh_cipsi_conv, max_iter=max_cipsi_iter)
        print("")
        print(" TPSCI:          %12.8f      Dim:%6d" % (e0+ecore, len(ci_vector)))
        print(" TPSCI(2):       %12.8f      Dim:%6d" % (e2+ecore,len(pt_vector)))
        print("--------         TPSCI no tucker converged        --------")
        print("\n\n")

        
    else:
        ci_vector, pt_vector, e0, e2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham, 
            thresh_cipsi, thresh_ci_clip, thresh_cipsi_conv, max_cipsi_iter, 
            thresh_tucker_conv, max_tucker_iter, tucker_state_clip, hshift)
        print("")
        print(" TPSCI:          %12.8f      Dim:%6d" % (e0+ecore, len(ci_vector)))
        print(" TPSCI(2):       %12.8f      Dim:%6d" % (e2+ecore,len(pt_vector)))
        if t_conv == True:
            print("--------         TPSCI converged        --------")
            print("\n\n")
        elif t_conv == False:
            print("--------         TPSCI did not converged        --------")
            print("\n\n")

    return ci_vector, pt_vector, e0+ecore, e2+ecore 
# }}}
