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
np.set_printoptions(suppress=True, precision=6, linewidth=1500)
print("GITHUB TREE")
import subprocess
label = subprocess.check_output(["git","rev-parse", "HEAD"]).strip()
print(label)

###     PYSCF INPUT
r0 = 2.0
molecule = '''
C  -6.780165   0.224843   0.000000
C   6.780165  -0.224843   0.000000
C  -5.587871  -0.395931   0.000000
C   5.587871   0.395931   0.000000
H  -6.855486   1.309093   0.000000
H   6.855486  -1.309093   0.000000
H  -7.711938  -0.330412   0.000000
H   7.711938   0.330412   0.000000
H  -5.558134  -1.485572   0.000000
H   5.558134   1.485572   0.000000
C  -4.310074   0.280869   0.000000
C   4.310074  -0.280869   0.000000
C  -3.110354  -0.354381   0.000000
C   3.110354   0.354381   0.000000
H  -4.329470   1.370910   0.000000
H   4.329470  -1.370910   0.000000
H  -3.097480  -1.444853   0.000000
H   3.097480   1.444853   0.000000
C  -1.837018   0.309854   0.000000
C   1.837018  -0.309854   0.000000
C  -0.636207  -0.330789   0.000000
C   0.636207   0.330789   0.000000
H  -1.847524   1.400093   0.000000
H   1.847524  -1.400093   0.000000
H  -0.626779  -1.421145   0.000000
H   0.626779   1.421145   0.000000
'''
charge = 0
spin  = 0
basis_set = 'sto-3g'

npoly = 6
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
#mf = scf.RHF(mol).run()
mf = scf.RHF(mol).run(conv_tol=1e-14,max_cycle=200)
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

def mulliken_ordering(mol,norb,C):
# {{{
    """
    pyscf mulliken
    """
    S = mol.intor('int1e_ovlp_sph')
    mulliken = np.zeros((mol.natm,norb))
    for i in range(0,norb):
        Cocc = C[:,i].reshape(C.shape[0],1)
        temp = Cocc @ Cocc.T @ S   
        for m,lb in enumerate(mol.ao_labels()):
            print(lb)
            v1,v2,v3 = lb.split()
            print(v1)
            mulliken[int(v1),i] += temp[m,m]
    print(mulliken)
    return mulliken
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
    idx  = np.argsort(np.diag(h))
    print(idx)
    h,g = reorder_integrals(idx,h,g)
    C = C[:,idx]
    print(h)
    molden.from_mo(mol, 'h8.molden', C)
    idx = e1_order(h,1e-1)
    h,g = reorder_integrals(idx,h,g)
    print(h)
    C = C[:,idx]
    molden.from_mo(mol, 'h8.molden', C)
    blocks = [range(0,4),range(4,8),range(8,12)]
    init_fspace = ((2, 2),(2, 2),(2, 2))

do_tci = 1
if do_tci:

    #clusters, clustered_ham, ci_vector = system_setup(h, g, ecore, blocks, init_fspace, 
    #                                                    cmf_maxiter     = 20,
    #                                                    cmf_dm_guess    = None,
    #                                                    cmf_diis        = False,
    #                                                    max_roots       = 100,
    #                                                    delta_elec      = 4
    #                                                    )

    #edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    #print("EDPS: %16.8f"%edps)

    from scipy import optimize   

    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C)
    oocmf.init()

    x = np.zeros_like(h)
    #x = oocmf.grad(x)

    #Gpq = -Gpq.ravel()


    #edps = oocmf.energy_dps()
    opt_result = scipy.optimize.minimize(oocmf.energy, x, jac=oocmf.grad, method = 'BFGS')
    exit()
    #opt_result = scipy.optimize.minimize(oocmf.energy, x, jac='3-point', method = 'BFGS')
    ##opt_result = scipy.optimize.fmin_cg(oocmf.energy, x)
    edps = oocmf.energy_dps()
    for git in range(0,20):
        grad = oocmf.grad(x)
        x = -np.linalg.norm(grad) * grad
        #if np.linalg.norm(grad) < 1e-2:
        #    x *= 5
        oocmf.rotate(x)
        e_curr = oocmf.cmf_energy()
        print("CurrCMF:%12.8f       Grad:%12.8f    dE:%10.6f"%(e_curr,np.linalg.norm(grad),e_curr-edps))
        if abs(e_curr - edps) < 1e-7:
            print("Converged E%16.8f"%e_curr)
            break
        else:
            edps = e_curr

    for git in range(0,40):
        grad = oocmf.grad(x)
        x = -np.linalg.norm(grad) * grad
        #if np.linalg.norm(grad) < 1e-2:
        #    x *= 5
        oocmf.rotate(x)
        e_curr = oocmf.cmf_energy()
        print("CurrCMF:%12.8f       Grad:%12.8f     dE:%10.6f"%(e_curr,np.linalg.norm(grad),e_curr-edps))
        if abs(np.linalg.norm(grad)) < 1e-5:
            print("Converged E%16.8f"%e_curr)
            break
        else:
            edps = e_curr

    exit()
