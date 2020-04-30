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
np.set_printoptions(suppress=True, precision=2, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

###     PYSCF INPUT
r0 = 2.0
molecule = '''
C      1.248860      3.535930      0.000000  
C      2.437770      2.849500      0.000000  
C      2.468240      1.425040      0.000000  
C      3.686630      0.686420      0.000000  
C      3.686630     -0.686420      0.000000  
C      2.468240     -1.425040      0.000000  
C      2.437770     -2.849500      0.000000  
C      1.248860     -3.535930      0.000000  
C      0.000000     -2.850080      0.000000  
C     -1.248860     -3.535930      0.000000  
C     -2.437770     -2.849500      0.000000  
C     -2.468240     -1.425040      0.000000  
C     -3.686630     -0.686420      0.000000  
C     -3.686630      0.686420      0.000000  
C     -2.468240      1.425040      0.000000  
C     -2.437770      2.849500      0.000000  
C     -1.248860      3.535930      0.000000  
C      0.000000      2.850080      0.000000  
C      0.000000      1.428120      0.000000  
C      1.236790      0.714060      0.000000  
C      1.236790     -0.714060      0.000000  
C      0.000000     -1.428120      0.000000  
C     -1.236790     -0.714060      0.000000  
C     -1.236790      0.714060      0.000000  
H      1.245540      4.623770      0.000000  
H      3.381530      3.390560      0.000000  
H      4.627070      1.233220      0.000000  
H      4.627070     -1.233220      0.000000  
H      3.381530     -3.390560      0.000000  
H      1.245540     -4.623770      0.000000  
H     -1.245540     -4.623770      0.000000  
H     -3.381530     -3.390560      0.000000  
H     -4.627070     -1.233220      0.000000  
H     -4.627070      1.233220      0.000000  
H     -3.381530      3.390560      0.000000  
H     -1.245540      4.623770      0.000000  
'''
charge = 0
spin  = 0
basis_set = 'sto-3g'

npoly = 12
na = npoly
nb = npoly

###     TPSCI BASIS INPUT
orb_basis = 'PM'
cas_nel = 2*npoly
cas_norb = 2*npoly

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

local = False
local = True
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

print("ecore %16.8f"%ecore)


if local:
    print("MULLIKEN")
    m1 = mulliken_ordering(mol,h.shape[0],C)

    ppp = np.column_stack(np.where(m1>.90))
    print(ppp)
    idx = list(ppp[:,1])
    print(m1.shape)
    print(m1)
    m2 = np.where(m1>.90)
    print(m2)
    print(m2[1])

    idx = m2[1]

    C = C[:,idx]
    h,g = reorder_integrals(idx,h,g)
    molden.from_mo(mol, 'h8.molden', C)

    blocks = [[0,1,2,17,18,19],[3,4],[5,6,7,8,20,21],[9,10],[11,12,13,14,22,23],[15,16]]
    init_fspace = ((3,3),(1,1),(3,3),(1,1),(3,3),(1,1))



do_tci = 1
if do_tci:

    clusters, clustered_ham, ci_vector = system_setup(h, g, ecore, blocks, init_fspace, 
                                                        cmf_maxiter     = 20,
                                                        cmf_dm_guess    = None,
                                                        cmf_diis        = False,
                                                        max_roots       = 100,
                                                        delta_elec      = 4
                                                        )

    ndata = 0
    for ci in clusters:
        for o in ci.ops:
            for f in ci.ops[o]:
                ndata += ci.ops[o][f].size * ci.ops[o][f].itemsize
    print(" Amount of data stored in TDMs: %12.2f Gb" %(ndata * 1e-9))

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)

    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'mp',
                        thresh_cipsi        = 1e-3,
                        thresh_ci_clip      = 1e-6,
                        max_tucker_iter     = 2,
                        nbody_limit         = 2,
                        thresh_search       = 1e-3,
                        thresh_asci         = 1e-2,
                        tucker_state_clip   = 100,  #don't use any pt for tucker 
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)

    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'mp',
                        thresh_cipsi        = 1e-5,
                        thresh_ci_clip      = 1e-7,
                        max_tucker_iter     = 2,
                        nbody_limit         = 4,
                        thresh_search       = 1e-4,
                        thresh_asci         = 1e-2,
                        tucker_state_clip   = 100,  #don't use any pt for tucker 
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)
    
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'mp',
                        thresh_cipsi        = 1e-6,
                        thresh_ci_clip      = 1e-8,
                        max_tucker_iter     = 2,
                        nbody_limit         = 4,
                        thresh_search       = 1e-4,
                        thresh_asci         = 1e-2,
                        tucker_state_clip   = 100,  #don't use any pt for tucker 
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)
    
    ci_vector, pt_vector, etci, etci2, t_conv = bc_cipsi_tucker(ci_vector.copy(), clustered_ham,
                        pt_type             = 'mp',
                        thresh_cipsi        = 1e-7,
                        thresh_ci_clip      = 1e-9,
                        max_tucker_iter     = 4,
                        nbody_limit         = 4,
                        thresh_search       = 1e-4,
                        thresh_asci         = 1e-2,
                        tucker_state_clip   = 100,  #don't use any pt for tucker 
                        tucker_conv_target  = 0,    #converge variational energy
                        nproc               = None)

    tci_dim = len(ci_vector)
    ci_vector.print()
    ecore = clustered_ham.core_energy

    etci += ecore
    etci2 += ecore


    print(" TCI:        %12.9f Dim:%6d"%(etci,tci_dim))

