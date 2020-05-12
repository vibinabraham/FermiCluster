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

np.set_printoptions(suppress=True, precision=3, linewidth=1500)

dim_a = 4
dim_b = 4
U = 8
beta1 = 1
beta2 = 0.2
n_orb = dim_a * dim_b

h, g = make_2d_lattice(dim_a,dim_b,beta1,beta2,U)


blocks = [[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]
init_fspace = ((2,2),(2,2),(2,2),(2,2))
nelec = tuple([sum(x) for x in zip(*init_fspace)])
#blocks = [[0,1],[2,3]]

C = np.eye(h.shape[0])

do_fci = 0
do_hci = 0
do_tci = 1

if do_fci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=0)
if do_hci:
    pyscf.lib.num_threads(4)  #with degenerate states and multiple processors there can be issues
    Escf,orb,h2,g2,C = run_hubbard_scf(h,g,n_orb//2)
    ehci, hci_dim = run_hci_pyscf(h2,g2,nelec,ecore=0,select_cutoff=1e-3,ci_cutoff=1e-3)
if do_tci:
    ecore = 0
    # Initialize the CMF solver. 
    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100)
    oocmf.init() # runs a single step CMF calculation

    oo = False
    if oo:
        x = np.zeros_like(h)
        min_options = {'gtol': 1e-8, 'disp':False}
        opt_result = scipy.optimize.minimize(oocmf.energy, x, jac=oocmf.grad, method = 'BFGS', options=min_options )
        #opt_result = scipy.optimize.minimize(oocmf.energy, x, jac=oocmf.grad, method = 'BFGS', callback=oocmf.callback)
        print(opt_result.x)
        Kpq = opt_result.x.reshape(h.shape)
        print(Kpq)

        e_fcmf = oocmf.energy_dps()
        oocmf.rotate(Kpq)
        e_ocmf = oocmf.energy_dps()
        print("Orbital Frozen    CMF:%12.8f"%(e_fcmf+ecore))
        print("Orbital Optimized CMF:%12.8f"%(e_ocmf+ecore))


    clustered_ham = oocmf.clustered_ham  # clustered_ham used for TPSCI calculation
    ci_vector = oocmf.ci_vector   # lowest energy TPS using the given Fock space

    e_ref = build_hamiltonian_diagonal(clustered_ham,ci_vector) # compute reference energy
    print("Reference TPS Energy:%16.10f"%(e_ref+ecore))

    # Run the TPSCI calculation on top of the CMF reference 
    ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'en',
                        thresh_cipsi        = 1e-6,
                        thresh_ci_clip      = 1e-7,
                        max_tucker_iter     = 3,
                        nbody_limit         = 4,
                        shared_mem          = 1e9,
                        thresh_search       = 1e-6,
                        thresh_asci         = 1e-2,
                        matvec=3,
                        tucker_state_clip   = 1e-6,
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)

    e2,pt_vector = compute_pt2_correction(ci_vector, clustered_ham, etci,
            thresh_asci     = 0,
            thresh_search   = 1e-8,
            pt_type         = 'en',
            matvec          = 3)


    tci_dim = len(ci_vector)
    ci_vector.print_configs()
    print(" TPSCI:          %12.8f      Dim:%6d" % (etci, len(ci_vector)))
    print(" TPSCI+PT:       %12.8f             " % (etci+e2))
    etci2 = etci+e2
    etci = etci

if do_fci:
    print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
if do_hci:
    print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
if do_tci:
    print(" TPSCI:      %12.8f Dim:%6d" % (etci, tci_dim))

