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
np.set_printoptions(suppress=True, precision=2, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues



def generate_hamiltonian():
    ###     PYSCF INPUT
    r0 = 1.50
    molecule = '''
    Cr
    Cr   1   {}
    '''.format(r0)
    charge = 0
    spin  = 0
    basis_set = 'def2-svp'

    ###     TPSCI BASIS INPUT
    orb_basis = 'scf'
    cas = True
    cas_nstart = 12
    cas_nstop =  42
    loc_start = 1
    loc_stop = 6
    cas_nel = 24

    def ordering_diatomics_cr(mol,C):
    # {{{
        ##DZ basis diatomics reordering with frozen 1s

        orb_type = ['s','pz','dz','px','dxz','py','dyz','dx2-y2','dxy']
        ref = np.zeros(C.shape[1])

        ## Find dimension of each space
        dim_orb = []
        for orb in orb_type:
            print("Orb type",orb)
            idx = 0
            for label in mol.ao_labels():
                if orb in label:
                    #print(label)
                    idx += 1

            ##frozen 1s orbitals
            if orb == 's':
                idx -= 6
            elif orb == 'px':
                idx -=2
            elif orb == 'py':
                idx -=2
            elif orb == 'pz':
                idx -=2
            dim_orb.append(idx)
            print(idx)


        new_idx = []
        ## Find orbitals corresponding to each orb space
        for i,orb in enumerate(orb_type):
            print("Orbital type:",orb)
            from pyscf import mo_mapping
            s_pop = mo_mapping.mo_comps(orb, mol, C)
            print(s_pop)
            ref += s_pop
            cas_list = s_pop.argsort()[-dim_orb[i]:]
            print('cas_list', np.array(cas_list))
            new_idx.extend(cas_list)
            #print(orb,' population for active space orbitals', s_pop[cas_list])

        ao_labels = mol.ao_labels()
        #idx = mol.search_ao_label(['N.*s'])
        #for i in idx:
        #    print(i, ao_labels[i])
        print(ref)
        print(new_idx)
        for label in mol.ao_labels():
            print(label)

        return new_idx
    # }}}

    # basis is SVP read comments by alex thom paper DOI:10.1021/acs.jctc.9b01023
    from pyscf import gto
    basis_set={'Cr': gto.parse('''
    BASIS "ao basis" PRINT
    #BASIS SET: (14s,8p,5d) -> [5s,2p,2d]
    Cr    S
      51528.086349               0.14405823106E-02
       7737.2103487              0.11036202287E-01
       1760.3748470              0.54676651806E-01
        496.87706544             0.18965038103
        161.46520598             0.38295412850
         55.466352268            0.29090050668
    Cr    S
        107.54732999            -0.10932281100
         12.408671897            0.64472599471
          5.0423628826           0.46262712560
    Cr    S
          8.5461640165          -0.22711013286
          1.3900441221           0.73301527591
          0.56066602876          0.44225565433
    Cr    S
          0.71483705972E-01      1.0000000000
    Cr    S
          0.28250687604E-01      1.0000000000
    Cr    P
        640.48536096             0.96126715203E-02
        150.69711194             0.70889834655E-01
         47.503755296            0.27065258990
         16.934120165            0.52437343414
          6.2409680590           0.34107994714
    Cr    P
          3.0885463206           0.33973986903
          1.1791047769           0.57272062927
          0.43369774432          0.24582728206
    Cr    D
         27.559479426            0.30612488044E-01
          7.4687020327           0.15593270944
          2.4345903574           0.36984421276
          0.78244754808          0.47071118077
    Cr    D
          0.21995774311          0.33941649889
    END''')}


    ###     TPSCI CLUSTER INPUT
    init_fspace = ((1, 1),(3, 3),(3, 3),(3, 3), (1, 1), (1, 1))
    blocks = [range(0,4),range(4,10),range(10,16),range(16,22),range(22,26),range(26,30)]

    # Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore
    print("Ecore:%16.8f"%ecore)
    C = pmol.C
    K = pmol.K
    mol = pmol.mol
    mo_energy = pmol.mf.mo_energy
    dm_aa = pmol.dm_aa
    dm_bb = pmol.dm_bb

    do_fci = 0
    do_hci = 0
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=5e-4,ci_cutoff=5e-4)

    #cluster using hcore

    idx = ordering_diatomics_cr(mol,C)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    mo_energy = mo_energy[idx]
    dm_aa = dm_aa[:,idx]
    dm_aa = dm_aa[idx,:]
    dm_bb = dm_bb[:,idx]
    dm_bb = dm_bb[idx,:]

    print(dm_aa)


    from pyscf import molden
    molden.from_mo(pmol.mol, 'h8.molden', C)
    print(h)
    mol = pmol.mol
    if mol.symmetry == True:
        from pyscf import symm
        mo = symm.symmetrize_orb(mol, C)
        osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
        for i in range(len(osym)):
            print("%4d %8s %16.8f"%(i+1,osym[i],mo_energy[i]))




    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    print(" Clusters:")
    [print(ci) for ci in clusters]

    clustered_ham = ClusteredOperator(clusters, core_energy=ecore)
    print(" Add 1-body terms")
    clustered_ham.add_local_terms()
    clustered_ham.add_1b_terms(h)
    print(" Add 2-body terms")
    clustered_ham.add_2b_terms(g)

    # intial state
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)
    ci_vector.print()

    # do cmf
    do_cmf = 1
    if do_cmf:
        # Get CMF reference
        e_cmf, cmf_conv, rdm_a, rdm_b = cmf(clustered_ham, ci_vector, h, g, max_iter=10, dm_guess=(dm_aa, dm_bb), diis=True)

    print(" Final CMF Total Energy %12.8f" %(e_cmf + ecore))

    # build cluster basis and operator matrices using CMF optimized density matrices
    for ci_idx, ci in enumerate(clusters):
        print(ci)
        fspaces_i = init_fspace[ci_idx]
        delta_e = 2
        fspaces_i = ci.possible_fockspaces( delta_elec=(fspaces_i[0], fspaces_i[1], delta_e) )


        print()
        print(" Form basis by diagonalizing local Hamiltonian for cluster: ",ci_idx)

        ci.form_fockspace_eigbasis(h, g, fspaces_i, max_roots=100, rdm1_a=rdm_a, rdm1_b=rdm_b)


        print(" Build mats for cluster ",ci.idx)
        ci.build_op_matrices()
        ci.build_local_terms(h,g)


    hamiltonian_file = open('hamiltonian_file', 'wb')
    pickle.dump(clustered_ham, hamiltonian_file)

    print(" Done.")

def run():
    hamiltonian_file = open('hamiltonian_file', 'rb')
    clustered_ham = pickle.load(hamiltonian_file)

    clusters = clustered_ham.clusters

    ndata = 0
    for ci in clusters:
        for o in ci.ops:
            for f in ci.ops[o]:
                ndata += ci.ops[o][f].size * ci.ops[o][f].itemsize
    print(" Amount of data stored in TDMs: %12.2f Gb" %(ndata * 1e-9))

    init_fspace = ((1, 1),(3, 3),(3, 3),(3, 3), (1, 1), (1, 1))
    ci_vector = ClusteredState(clusters)
    ci_vector.init(init_fspace)

    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
                        pt_type         = 'mp',
                        thresh_cipsi    = 1e-3,
                        thresh_ci_clip  = 1e-5,
                        max_tucker_iter = 4,
                        thresh_asci     = 1e-2,
                        nbody_limit     = 4,
                        thresh_search   = 1e-5,
                        nproc           = None)

    tci_dim = len(ci_vector)
    ci_vector.print()
    ecore = clustered_ham.core_energy

    etci += ecore
    etci2 += ecore


    print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))


if __name__== "__main__":
    generate_hamiltonian()
    run()
