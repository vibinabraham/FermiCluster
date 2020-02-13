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
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

###     PYSCF INPUT
r0 = 2.0
molecule = '''
C   -1.244593000    1.402439000    0.000000000
C   -2.433051000    0.708272000    0.000000000
C   -2.433051000   -0.708272000    0.000000000
C   -1.244593000   -1.402439000    0.000000000
C    0.000000000   -0.716921000    0.000000000
C    0.000000000    0.716921000    0.000000000
H   -1.242226000    2.489450000    0.000000000
H   -3.377109000    1.245176000    0.000000000
H   -3.377109000   -1.245176000    0.000000000
H   -1.242226000   -2.489450000    0.000000000
C    1.244593000    1.402439000    0.000000000
C    2.433051000    0.708272000    0.000000000
H    3.377109000    1.245176000    0.000000000
C    2.433051000   -0.708272000    0.000000000
H    3.377109000   -1.245176000    0.000000000
C    1.244593000   -1.402439000    0.000000000
H    1.242226000   -2.489450000    0.000000000
H    1.242226000    2.489450000    0.000000000
'''
charge = 0
spin  = 0
basis_set = 'sto-3g'

###     TPSCI BASIS INPUT
orb_basis = 'PM'
cas = False
cas_nstart = 10
cas_nstop =  58
cas_nel = 10
cas_norb = 10

na = 5
nb = 5

#Integrals from pyscf
import pyscf
from pyscf import gto, scf, ao2mo, molden, lo
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
#PYSCF inputs

mol = gto.Mole()
mol.atom = molecule

mol.max_memory = 1000 # MB
mol.symmetry = True
mol.charge = charge
mol.spin = spin
mol.basis = basis_set
mol.build()
print("symmertry")
print(mol.topgroup)

#SCF

#mf = scf.RHF(mol).run(init_guess='atom')
mf = scf.RHF(mol).run()
#C = mf.mo_coeff #MO coeffs
enu = mf.energy_nuc()

if mol.symmetry == True:
    from pyscf import symm
    mo = symm.symmetrize_orb(mol, mf.mo_coeff)
    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    for i in range(len(osym)):
        print("%4d %8s %16.8f"%(i+1,osym[i],mf.mo_energy[i]))

mo_occ = mf.mo_occ>0
mo_vir = mf.mo_occ==0
print(mo_occ)
print(mo_vir)
mo_occ = np.where(mf.mo_occ>0)[0]
mo_vir = np.where(mf.mo_occ==0)[0]
print(mo_occ)
print(mo_vir)

from pyscf import mo_mapping
s_pop = mo_mapping.mo_comps('C 2pz', mol, mf.mo_coeff)
print(s_pop)
cas_list = s_pop.argsort()[-cas_norb:]
print('cas_list', np.array(cas_list))
print('s population for active space orbitals', s_pop[cas_list])


focc_list = list(set(mo_occ)-set(cas_list))
print(focc_list)
fvir_list = list(set(mo_vir)-set(cas_list))
print(fvir_list)


def get_eff_for_casci(focc_list,cas_list,h,g):
# {{{

    cas_dim = len(cas_list)
    const = 0
    for i in focc_list:
        const += 2 * h[i,i]
        for j in focc_list:
            const += 2 * g[i,i,j,j] -  g[i,j,i,j]

    eff = np.zeros((cas_dim,cas_dim))
    for L,l in enumerate(cas_list):
        for M,m in enumerate(cas_list):
            for j in focc_list:
                eff[L,M] += 2 * g[l,m,j,j] -  g[l,j,j,m]
    return const, eff
# }}}

def reorder_integrals(idx,h,g):
# {{{
    h = h[:,idx]
    h = h[idx,:]

    g = g[:,:,:,idx]
    g = g[:,:,idx,:]
    g = g[:,idx,:,:]
    g = g[idx,:,:,:]
    return h,g
# }}}

local = True
local = False
if local:
    cl_c = mf.mo_coeff[:, focc_list]
    cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_list]).kernel(verbose=4)
    cl_v = mf.mo_coeff[:, fvir_list]
    C = np.column_stack((cl_c, cl_a, cl_v))
else:
    cl_c = mf.mo_coeff[:, focc_list]
    cl_a = mf.mo_coeff[:, cas_list]
    cl_v = mf.mo_coeff[:, fvir_list]
    C = np.column_stack((cl_c, cl_a, cl_v))

if mol.symmetry == True:
    orb_energy = mf.mo_energy
    orb_energy = orb_energy[cas_list]
    from pyscf import symm
    mo = symm.symmetrize_orb(mol, cl_a)
    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
    #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
    for i in range(len(osym)):
        print("%4d %8s %16.8f"%(i+1,osym[i],orb_energy[i]))

from pyscf import mo_mapping
s_pop = mo_mapping.mo_comps('C 2pz', mol, C)
print(s_pop)
cas_list = s_pop.argsort()[-cas_norb:]
print('cas_list', np.array(cas_list))
print('s population for active space orbitals', s_pop[cas_list])
focc_list = list(set(mo_occ)-set(cas_list))
print(focc_list)
fvir_list = list(set(mo_vir)-set(cas_list))
print(fvir_list)



h = C.T.dot(mf.get_hcore()).dot(C)
g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((h.shape[0]),))
const,eff = get_eff_for_casci(focc_list,cas_list,h,g)


focc_list = list(set(mo_occ)-set(cas_list))
print(focc_list)
fvir_list = list(set(mo_vir)-set(cas_list))
print(fvir_list)

ecore = enu + const
h,g = reorder_integrals(cas_list,h,g)
h = h + eff
C = C[:,cas_list]
molden.from_mo(mol, 'h8.molden', C)


idx  = np.argsort(np.diag(h))
print(idx)
h,g = reorder_integrals(idx,h,g)
C = C[:,idx]

do_hci2 = 0
do_hci = 0
do_fci = 0


if do_fci:
    from pyscf import fci
    #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE
    cisolver = fci.direct_spin1.FCI()
    efci, ci = cisolver.kernel(h, g, h.shape[1], nelec=(na,nb), ecore=ecore,nroots =1,verbose=100)
    #d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
    #print(d1)
    print("FCIS%10.8f"%(efci))

    cisolver = fci.direct_spin1.FCI()
    efci2, ci = cisolver.kernel(h, g, h.shape[1], nelec=(na+1,nb-1), ecore=ecore,nroots =1,verbose=100)
    fci_dim = ci.shape[0] * ci.shape[1]

    print("FCIT%10.8f"%(efci2))
    print("%10.6f"%(627.5*(efci2-efci)))
    fcist = efci2-efci
    print("FCI IP: %10.8f"%(27.2114*(fcist)))

if do_hci:
    from pyscf.hci import hci
    cisolver = hci.SCI()
    cisolver.select_cutoff = 1e-3
    cisolver.ci_coeff_cutoff = 1e-3
    cisolver.maxiter = 20
    ehci, civec = cisolver.kernel(h, g, h.shape[1], (na,nb), ecore=ecore,verbose=4,nroots=4)
    hci_dim = civec[0].shape[0]
    print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
    #import random
    #idx = random.sample(range(0,10), 10)
    #h,g = reorder_integrals(idx,h,g)
    #C = C[:,idx]
    #ehcis, hci_dim = run_hci_pyscf(h,g,(na,nb),ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)
    ehcit, hci_dim = run_hci_pyscf(h,g,(na+1,nb-1),ecore=ecore,select_cutoff=1e-10,ci_cutoff=1e-10)
    print(627.5*(ehcit-ehcis))
    print("HCI ST: %10.8f"%(27.2114*(ehcit-ehcis)))

if do_hci2:
    from pyscf.hci import hci
    h1 = h
    eri = g
    norb = 10
    nelec = 6,4

    hf_str = np.hstack([hci.orblst2str(range(nelec[0]), norb),
                       hci.orblst2str(range(nelec[1]), norb)]).reshape(1,-1)

    eri_sorted = abs(eri).argsort()[::-1]
    jk = eri.reshape([norb]*4)
    jk = jk - jk.transpose(2,1,0,3)
    jk = jk.ravel()
    jk_sorted = abs(jk).argsort()[::-1]
    ci1 = [hci.as_SCIvector(np.ones(1), hf_str)]

    myci = hci.SelectedCI()
    myci.select_cutoff = .001
    myci.ci_coeff_cutoff = .001

    ci2 = hci.enlarge_space(myci, ci1, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)
    print("CI2 kya hai")
    print(len(ci2[0]))

    ci2 = hci.enlarge_space(myci, ci1, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)
    np.random.seed(1)
    ci3 = np.random.random(ci2[0].size)
    ci3 *= 1./np.linalg.norm(ci3)
    ci3 = [ci3]
    ci3 = hci.enlarge_space(myci, ci2, h1, eri, jk, eri_sorted, jk_sorted, norb, nelec)

    efci = hci.direct_spin1.kernel(h1, eri, norb, nelec, verbose=4)[0]

    ci4 = hci.contract_2e_ctypes((h1, eri), ci3[0], norb, nelec)
    print(ci4.shape)
    print(len(ci3))
    for i in range(len(ci4)):
        ci3[0][i] = ci4[i]

    fci3 = hci.to_fci(ci3, norb, nelec)
    h2e = hci.direct_spin1.absorb_h1e(h1, eri, norb, nelec, .5)
    fci4 = hci.direct_spin1.contract_2e(h2e, fci3, norb, nelec)
    fci4 = hci.from_fci(fci4, ci3[0]._strs, norb, nelec)
    print("sum kay hai")
    print(abs(ci4-fci4).sum())
    for i in range(len(ci4)):
        ci3[0][i] = fci4[i]
    print(ci4)
    print(fci4)

    print(ci3)
    print(len(ci3))
    e = myci.kernel(h1, eri, norb, nelec, verbose=4,ci0=ci3)[0]
    print(e, efci)

if local:

    #import random
    #idx = random.sample(range(0,10), 10)
    #h,g = reorder_integrals(idx,h,g)
    #C = C[:,idx]

    idx = e1_order(h,1e-1)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    print(h)
    molden.from_mo(mol, 'h8.molden', C)
    blocks = [[0,1,2,5],[4,7],[3,6,8,9]]
    init_fspace = ((2, 2),(1, 1),(2, 2))

    #blocks = [[0,1,2],[5,4,7,9],[3,6,8]]
    #init_fspace = ((2, 1),(2, 2),(1, 2))

else:
    idx  = np.argsort(np.diag(h))
    print(idx)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    molden.from_mo(mol, 'h8.molden', C)
    print(h)
    blocks = [range(0,3),range(3,7),range(7,10)]
    init_fspace = ((3, 3),(2, 2),(0, 0))
    blocks = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    init_fspace = ((1, 1),(1, 1),(1, 1),(1, 1),(1, 0),(1, 0),(0, 0),(0, 0),(0, 0),(0, 0))

    """
    ##RCM order
    idx = e1_order(h,1e-1)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    print(h)
    blocks = [range(0,3),range(3,6),range(6,10)]
    init_fspace = ((1, 1),(2, 2),(2, 2))
    if mol.symmetry == True:
        from pyscf import symm
        mo = symm.symmetrize_orb(mol, C)
        osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
        for i in range(len(osym)):
            print("%4d %8s %16.8f"%(i+1,osym[i],h[i,i]))
    """



do_tci = 0
do_tci2 = 1
if do_tci:
    #ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
    #    thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=20,max_cipsi_iter=20)
    #ci_vector.print_configs()
    #tci_dim = len(ci_vector)
    max_cipsi_iter=20
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

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-3, thresh_ci_clip=0, max_tucker_iter=3,asci_clip=0)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-4, thresh_ci_clip=5e-2, max_tucker_iter=3,asci_clip=0)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-5, thresh_ci_clip=1e-3, max_tucker_iter=3,asci_clip=0)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-6, thresh_ci_clip=5e-4, max_tucker_iter=3,asci_clip=0)
    print("init DPS ",(edps+ecore))
    print("")
    print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
    print(" TPSCI(2):       %12.8f      Dim:%6d" % (etci2+ecore,len(pt_vector)))
    print("coefficient of dominant determinant")
    ci_vector.print_configs()
    tci_dim = len(ci_vector)
    etci = etci+ecore
    etci2 = etci2+ecore

    etcis = etci
    etcispt = etci2

if do_tci2:
    #ci_vector, pt_vector, etci, etci2 = run_tpsci(h,g,blocks,init_fspace,ecore=ecore,
    #    thresh_ci_clip=1e-3,thresh_cipsi=1e-6,max_tucker_iter=20,max_cipsi_iter=20)
    #ci_vector.print_configs()
    #tci_dim = len(ci_vector)
    max_cipsi_iter=20
    n_blocks = len(blocks)

    clusters = []

    for ci,c in enumerate(blocks):
        clusters.append(Cluster(ci,c))

    ci_vector = ClusteredState(clusters)
    ## RCM
    #ci_vector.init(((2, 0),(2, 2),(2, 2)))
    #ci_vector.init(((1, 1),(3, 1),(2, 2)))
    #ci_vector.init(((1, 1),(2, 2),(3, 1)))
    ## LOCAL
    #ci_vector.init(((3, 1),(1, 1),(2, 2)))
    #ci_vector.init(((2, 2),(2, 0),(2, 2)))
    #ci_vector.init(((2, 2),(1, 1),(3, 1)))
    ## Energy
    #ci_vector.init(((3, 3),(3, 1),(0, 0)))

    ## LOCAL cl2
    #ci_vector.init(((2, 1),(3, 1),(1, 2)))

    ci_vector.init(((1, 1),(1, 1),(1, 1),(1, 1),(1, 0),(1, 0),(0, 0),(0, 0),(0, 0),(0, 0)))
    ci_vector.init(((1, 1),(1, 1),(1, 1),(1, 1),(0, 0),(0, 0),(1, 0),(1, 0),(0, 0),(0, 0)))
    ci_vector.init(((1, 1),(1, 1),(0, 1),(0, 1),(1, 0),(1, 0),(1, 0),(1, 0),(0, 0),(0, 0)))
    ci_vector.init(((0, 1),(0, 1),(1, 1),(1, 1),(1, 0),(1, 0),(1, 0),(1, 0),(0, 0),(0, 0)))
    ci_vector.init(((1, 1),(1, 1),(1, 1),(1, 1),(0, 0),(0, 0),(0, 0),(1, 0),(1, 0),(0, 0)))
    ci_vector.init(((1, 1),(1, 1),(0, 1),(0, 1),(1, 0),(1, 0),(0, 0),(1, 0),(1, 0),(0, 0)))
    ci_vector.init(((0, 1),(0, 1),(1, 1),(1, 1),(1, 0),(1, 0),(0, 0),(1, 0),(1, 0),(0, 0)))
    ci_vector.init(((1, 1),(1, 1),(1, 1),(1, 1),(0, 0),(0, 0),(0, 0),(0, 0),(1, 0),(1, 0)))
    ci_vector.init(((1, 1),(1, 1),(0, 1),(0, 1),(1, 0),(1, 0),(0, 0),(0, 0),(1, 0),(1, 0)))
    ci_vector.init(((0, 1),(0, 1),(1, 1),(1, 1),(1, 0),(1, 0),(0, 0),(0, 0),(1, 0),(1, 0)))
    ci_vector.init(((0, 0),(0, 0),(1, 1),(1, 1),(1, 0),(1, 0),(0, 1),(0, 1),(1, 0),(1, 0)))


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

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    #ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-3, thresh_ci_clip=0, max_tucker_iter=3,asci_clip=0)
    #ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-4, thresh_ci_clip=5e-2, max_tucker_iter=3,asci_clip=0)
    #ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-5, thresh_ci_clip=1e-3, max_tucker_iter=3,asci_clip=0)
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,thresh_cipsi=1e-6, thresh_ci_clip=5e-4, max_tucker_iter=3,asci_clip=0)

    #ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
    #    thresh_cipsi=1e-4, thresh_ci_clip=5e-3, max_tucker_iter=3,asci_clip=.1)
    print("init DPS",(edps+ecore))
    print("")
    print(" TPSCI:          %12.8f      Dim:%6d" % (etci+ecore, len(ci_vector)))
    print(" TPSCI(2):       %12.8f      Dim:%6d" % (etci2+ecore,len(pt_vector)))
    print("coefficient of dominant determinant")
    ci_vector.print_configs()
    tci_dim = len(ci_vector)
    etci = etci+ecore
    etci2 = etci2+ecore

etcit = etci
etcitpt = etci2
print("ST: %10.8f"%(627.5*(etcit-etcis)))
print("ST: %10.8f"%(627.5*(etcitpt-etcispt)))
if do_fci:
    print("FCI ST: %10.8f"%(627.5*(fcist)))
if do_hci:
    print("HCI ST: %10.8f"%(627.5*(ehcit-ehcis)))
    print("HCI dim",hci_dim)

print("TCI ST: %10.8f"%(27.2114*(etcit-etcis)))
print("TCI ST: %10.8f"%(27.2114*(etcitpt-etcispt)))
if do_fci:
    print("FCI ST: %10.8f"%(27.2114*(fcist)))
if do_hci:
    print("HCI ST: %10.8f"%(27.2114*(ehcit-ehcis)))
