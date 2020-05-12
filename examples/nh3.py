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
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues

for ri in range(0,26):
    ###     PYSCF INPUT
    r0 = 0.95 + 0.05 * ri
    molecule = '''
    N   {1}  {1}   {1} 
    H   {0}   {0}   0
    H   0   {0}   {0}
    H   {0}   0   {0}
    '''.format(r0,r0/2)
    r0 = np.sqrt(r0*r0*3/4)
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'ibmo'
    cas = True
    cas_nstart = 1
    cas_nstop =  8
    loc_start = 1
    loc_stop = 8
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    C = pmol.C

    do_fci = 1
    do_hci = 1
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

    #cluster using hcore
    idx = e1_order(h,cut_off = 2e-1)
    h,g = reorder_integrals(idx,h,g)
    if do_tci:
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
        print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
        print(" TPSCI+PT:       %12.8f             " % (etci+e2+ecore))
        etci2 = etci+e2+ecore
        etci = etci+ecore


    print("  rad      FCI          Dim          HCI       Dim          TPSCI      Dim       TPSCI(2)")
    print(" %4.2f  %12.9f   %6d     %12.9f  %6d %12.9f %6d %12.9f"%(r0,efci,fci_dim,ehci,hci_dim,etci,tci_dim,etci2))
    exit()
