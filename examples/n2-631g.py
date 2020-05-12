import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

import pickle

from tpsci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=3, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)


# set memory requirements
numpy_memory = 2

for ri in range(0,15):
    ###     PYSCF INPUT
    r0 = 1.2 + 0.1*ri
    molecule = '''
    N   0  0   0
    N   0  0   {}
    '''.format(r0)
    charge = 0
    spin  = 0
    basis_set = '6-31g'
    #basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 2
    cas_nstop =  18
    cas_nel = 10


    ###     TPSCI CLUSTER INPUT
    init_fspace = ((2, 2),(1, 1),(1, 1), (1, 1))
    blocks = [range(0,4),range(4,8),range(8,12),range(12,16)]

    #init_fspace = ((3, 3),(2, 2))
    #blocks = [range(0,4),range(4,8)]

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True)
                    #loc_nstart=loc_start,loc_nstop = loc_stop)

    C = pmol.C
    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    print("Ecore:%16.8f"%ecore)
    mol = pmol.mol
    mf = pmol.mf
    mo_energy = mf.mo_energy[cas_nstart:cas_nstop]
    dm_aa = pmol.dm_aa
    dm_bb = pmol.dm_bb

    do_fci = 0
    do_hci = 0
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
        exit()
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=5e-4,ci_cutoff=5e-4)


    #idx = e1_order(h,1e-2)
    idx = ordering_diatomics(mol,C,basis_set=basis_set)
    norb = cas_nstop - cas_nstart
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    mo_energy = mo_energy[idx]
    dm_aa = dm_aa[:,idx]
    dm_aa = dm_aa[idx,:]
    dm_bb = dm_bb[:,idx]
    dm_bb = dm_bb[idx,:]

    print(h)
    print(dm_aa)


    from pyscf import symm
    mo = symm.symmetrize_orb(mol, C)
    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    for i in range(len(osym)):
        print("%4d %8s %16.8f"%(i,osym[i],mo_energy[i]))
    print("r:",r0)
    print(h)


    if do_tci:

        oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C,max_roots=100)
        oocmf.init((dm_aa,dm_bb))

        oo =False
        if oo:
            from scipy import optimize

            #don't allow this to mix with each other
            #oocmf.freeze_cluster_mixing(0,(3,4))

            x = np.zeros_like(h)
            #edps = oocmf.energy_dps()
            min_options = {'gtol': 1e-8, 'disp':False}
            opt_result = scipy.optimize.minimize(oocmf.energy, x, jac=oocmf.grad, method = 'BFGS', callback=oocmf.callback, options=min_options)
            print(opt_result.x)
            Kpq = opt_result.x.reshape(h.shape)

            e_fcmf = oocmf.energy_dps()
            oocmf.rotate(Kpq)
            e_ocmf = oocmf.energy_dps()
            print("Orbital Optimized CMF:%12.8f"%e_ocmf)

            from pyscf import molden
            molden.from_mo(mol, 'clustering_9.hf.molden', C)
            C = oocmf.C
            molden.from_mo(mol, 'clustering_9.cmf.molden', C)

            print(Kpq)

            print("Orbital Frozen    CMF:%12.8f"%e_fcmf)
            print("Orbital Optimized CMF:%12.8f"%e_ocmf)


        h = oocmf.h
        g = oocmf.g

        clustered_ham = oocmf.clustered_ham
        ci_vector = oocmf.ci_vector


        #filename = open('hamiltonian_file_test', 'wb')
        #pickle.dump(clustered_ham, filename)
        #filename.close()
         
        #filename = open('state_file_test', 'wb')
        #pickle.dump(ci_vector, filename)
        #filename.close()

        ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                            pt_type             = 'mp',
                            thresh_cipsi        = 1e-5,
                            thresh_ci_clip      = 1e-6,
                            max_tucker_iter     = 1,
                            nbody_limit         = 4,
                            shared_mem          = 1e9,
                            thresh_search       = 1e-6,
                            thresh_asci         = 1e-2,
                            matvec=3,
                            tucker_state_clip   = None,
                            tucker_conv_target  = 0,    #converge variational energy
                            nproc               = None)


        #filename = open('hamiltonian_file_test2', 'wb')
        #pickle.dump(clustered_ham, filename)
        #filename.close()
         
        #filename = open('state_file_test2', 'wb')
        #pickle.dump(ci_vector, filename)
        #filename.close()

        ci_vector, pt_vector, etci, etci2, t_conv = tpsci_tucker(ci_vector.copy(), clustered_ham,
                            pt_type             = 'mp',
                            thresh_cipsi        = 1e-7,
                            thresh_ci_clip      = 1e-8,
                            max_tucker_iter     = 1,
                            nbody_limit         = 4,
                            shared_mem          = 1e9,
                            matvec=3,
                            thresh_search       = 1e-7,
                            thresh_asci         = 1e-3,
                            tucker_state_clip   = None,
                            tucker_conv_target  = 0,    #converge variational energy
                            nproc               = None)

        #filename = open('hamiltonian_file_test3', 'wb')
        #pickle.dump(clustered_ham, filename)
        #filename.close()

        #filename = open('state_file_test3', 'wb')
        #pickle.dump(ci_vector, filename)
        #filename.close()


        e2a, vec2 = compute_pt2_correction(ci_vector, clustered_ham, etci,
                pt_type='mp',
                matvec=3,
                thresh_search=1e-8)

        e2b, vec2 = compute_pt2_correction(ci_vector, clustered_ham, etci,
                pt_type='en',
                matvec=3,
                thresh_search=1e-8)

        print("TPSCI-MPPT%16.8f"%(etci+ecore+e2a))
        print("TPSCI-ENPT%16.8f"%(etci+ecore+e2b))

