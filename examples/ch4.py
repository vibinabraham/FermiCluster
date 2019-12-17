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

for ri in range(0,20):
    ###     PYSCF INPUT
    r0 = 1.00 + 0.05 * ri
    molecule = '''
    C   {1}  {1}  {1} 
    H   {0}  {0}   0
    H    0   {0}  {0}
    H   {0}   0   {0}
    H    0    0    0
    '''.format(r0,r0/2)
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'ibmo'
    cas = True
    cas_nstart = 1
    cas_nstop =  9
    loc_start = 1
    loc_stop = 9
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6,7]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

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
        ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
            thresh_ci_clip=1e-6,thresh_cipsi=1e-6,max_tucker_iter=3,max_cipsi_iter=10)
        ci_vector.print_configs()
        tci_dim = len(ci_vector)

    print(h)
    print("  rad      FCI          Dim          HCI       Dim          TPSCI      Dim       TPSCI(2)")
    print(" %4.2f  %12.9f   %6d     %12.9f  %6d %12.9f %6d %12.9f"%(r0,efci,fci_dim,ehci,hci_dim,etci,tci_dim,etci2))
    exit()

