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

import dask
from dask.distributed import Client, progress

def bc_cipsi(ci_vector, clustered_ham, thresh_cipsi=1e-4, thresh_ci_clip=1e-5, thresh_conv=1e-8, max_iter=30):

    #client = Client(processes=False)
    client = Client(processes=True)

    pt_vector = ci_vector.copy()
    Hd_vector = ClusteredState(ci_vector.clusters)
    e_prev = 0
    for it in range(max_iter):
        print()
        print(" ===================================================================")
        print("     Selected CI Iteration: %4i epsilon: %12.8f" %(it,thresh_cipsi))
        print(" ===================================================================")
        print(" Build full Hamiltonian",flush=True)
        H = build_full_hamiltonian(clustered_ham, ci_vector)

        print(" Diagonalize Hamiltonian Matrix:",flush=True)
        e,v = np.linalg.eigh(H)
        idx = e.argsort()
        e = e[idx]
        v = v[:,idx]
        v0 = v[:,0]
        e0 = e[0]
        print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))

        ci_vector.zero()
        ci_vector.set_vector(v0)

        old_dim = len(ci_vector)

        if thresh_ci_clip > 0:
            print(" Clip CI Vector: thresh = ", thresh_ci_clip)
            print(" Old CI Dim: ", len(ci_vector))
            kept_indices = ci_vector.clip(thresh_ci_clip)
            print(" New CI Dim: ", len(ci_vector))
            if len(ci_vector) < old_dim:
                H = H[:,kept_indices][kept_indices,:]
                print(" Diagonalize Clipped Hamiltonian Matrix:",flush=True)
                e,v = np.linalg.eigh(H)
                idx = e.argsort()
                e = e[idx]
                v = v[:,idx]
                v0 = v[:,0]
                e0 = e[0]
                print(" Ground state of CI:                 %12.8f  CI Dim: %4i "%(e[0].real,len(ci_vector)))

                ci_vector.zero()
                ci_vector.set_vector(v0)


        for i,j,k in ci_vector:
            print(" iterator:", " Fock Space:", i, " Config:", j, " Coeff: %12.8f"%k)


        print(" Compute Matrix Vector Product:", flush=True)
        pt_vector = matvec1(clustered_ham, ci_vector)
        #pt_vector.print()


        var = pt_vector.norm() - e0*e0
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
        #Hd = build_hamiltonian_diagonal(clustered_ham, pt_vector, client)
        Hd = update_hamiltonian_diagonal(clustered_ham, pt_vector, Hd_vector)
        end = time.time()
        print(" Time spent in demonimator: ", end - start)

        denom = 1/(e0 - Hd)
        pt_vector_v = pt_vector.get_vector()
        pt_vector_v.shape = (pt_vector_v.shape[0])

        e2 = np.multiply(denom,pt_vector_v)
        pt_vector.set_vector(e2)
        e2 = np.dot(pt_vector_v,e2)

        print(" PT2 Energy Correction = %12.8f" %e2)
        print(" PT2 Energy Total      = %12.8f" %(e0+e2))

        print(" Choose which states to add to CI space")

        for fockspace,configs in pt_vector.items():
            for config,coeff in configs.items():
                if coeff*coeff > thresh_cipsi:
                    if fockspace in ci_vector:
                        ci_vector[fockspace][config] = 0
                    else:
                        ci_vector.add_fockspace(fockspace)
                        ci_vector[fockspace][config] = 0
        delta_e = e0 - e_prev
        e_prev = e0
        if len(ci_vector) <= old_dim and abs(delta_e) < thresh_conv:
            print(" Converged")
            break
        print(" Next iteration CI space dimension", len(ci_vector))
    #    print(" Do CMF:")
    #    for ci_idx, ci in enumerate(clusters):
    #        assert(ci_idx == ci.idx)
    #        print(" Extract local operator for cluster",ci.idx)
    #        opi = build_effective_operator(ci_idx, clustered_ham, ci_vector)
    #        print(" Form basis by diagonalize local Hamiltonian for cluster: ",ci_idx)
    #        ci.form_eigbasis_from_local_operator(opi,max_roots=1000)
    #        exit()
    client.close()
    return ci_vector, pt_vector, e0, e0+e2
