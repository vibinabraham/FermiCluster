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

def test_1():
    ###     PYSCF INPUT
    r0 = 2.0
    molecule = '''
    C  -4.308669   0.197146   0.000000
    C   4.308669  -0.197146   0.000000
    C  -3.110874  -0.411353   0.000000
    C   3.110874   0.411353   0.000000
    H  -4.394907   1.280613   0.000000
    H   4.394907  -1.280613   0.000000
    H  -5.234940  -0.367304   0.000000
    H   5.234940   0.367304   0.000000
    H  -3.069439  -1.500574   0.000000
    H   3.069439   1.500574   0.000000
    C  -1.839087   0.279751   0.000000
    C   1.839087  -0.279751   0.000000
    C  -0.634371  -0.341144   0.000000
    C   0.634371   0.341144   0.000000
    H  -1.871161   1.369551   0.000000
    H   1.871161  -1.369551   0.000000
    H  -0.607249  -1.431263   0.000000
    H   0.607249   1.431263   0.000000
    '''
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    npoly = 4
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
    mf = scf.RHF(mol).run(conv_tol=1e-14)
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
    #molden.from_mo(mol, 'oocmf_0.molden', C)

    print("ecore %16.8f"%ecore)

    print("MULLIKEN")
    m1 = mulliken_ordering(mol,h.shape[0],C)

    ppp = np.column_stack(np.where(m1>.90))
    print(ppp)
    idx = list(ppp[:,1])
    m2 = np.where(m1>.90)
    idx = m2[1]
    C = C[:,idx]
    h,g = reorder_integrals(idx,h,g)
    molden.from_mo(mol, 'h8.molden', C)

    blocks = [[0,1,2,3],[4,5],[6,7]]
    init_fspace = ((2,2),(1,1),(1,1))
    #blocks = [[0,1,2,3],[4,5,6,7]]
    #init_fspace = ((2,2),(2,2))


    from scipy import optimize

    oocmf = CmfSolver(h, g, ecore, blocks, init_fspace,C)
    oocmf.init()

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
    print("Orbital Frozen    CMF:%12.8f"%e_fcmf)
    print("Orbital Optimized CMF:%12.8f"%e_ocmf)

    h = oocmf.h
    g = oocmf.g
    clustered_ham = oocmf.clustered_ham
    ci_vector = oocmf.ci_vector

    edps = build_hamiltonian_diagonal(clustered_ham,ci_vector)
    print("%16.10f"%edps)

    #ref_grad = np.array([[-0.,      -0.,      -0.,      -0.,      -0.136637, -0.001116, -0.00094,   0.030398],
    #            [ 0.,       0.,      -0.,      -0.,      -0.001116, -0.136637,  0.030398, -0.00094 ],
    #            [ 0.,       0.,       0.,       0.,       0.001942, -0.033806,  0.123316, -0.002399],
    #            [ 0.,       0.,      -0.,      -0.,      -0.033806,  0.001942, -0.002399,  0.123316],
    #            [ 0.136637,  0.001116, -0.001942,  0.033806,  0.,       0.,       0.,       0.      ],
    #            [ 0.001116,  0.136637,  0.033806, -0.001942,  0.,      -0.,       0.,       0.      ],
    #            [ 0.00094,  -0.030398, -0.123316,  0.002399, -0.,      -0.,       0.,       0.      ],
    #            [-0.030398,  0.00094,   0.002399, -0.123316, -0.,      -0.,       0.,       0.      ]])

    ref_angles= np.array([[ 0.,      -0.,       -0.,       -0.,       -0.097782,  0.002467, -0.005287,  0.00192 ],
                         [ 0.,       -0.,       -0.,       -0.,        0.002467, -0.097782,  0.00192,  -0.005287],
                         [ 0.,        0.,        0.,        0.,       -0.002466, -0.123235,  0.155944, -0.004683],
                         [ 0.,        0.,       -0.,        0.,       -0.123235, -0.002466, -0.004683,  0.155944],
                         [ 0.097782, -0.002467,  0.002466,  0.123235,  0.,        0.,       -0.002248,  0.581917],
                         [-0.002467,  0.097782,  0.123235,  0.002466, -0.,       -0.,        0.581917, -0.002248],
                         [ 0.005287, -0.00192,  -0.155944,  0.004683,  0.002248, -0.581917,  0.,       -0.      ],
                         [-0.00192,   0.005287,  0.004683, -0.155944, -0.581917,  0.002248,  0.,        0.      ]])
    print(Kpq)                                                  
    print(ref_angles)
    try:
        assert(np.allclose(Kpq,ref_angles,atol=1e-5))
    except:
        assert(np.allclose(-1*Kpq,ref_angles,atol=1e-5))

    assert(abs(e_fcmf - -8.266997040181 ) <1e-8)
    assert(abs(e_ocmf - -8.528879972678 ) <1e-8)

if __name__== "__main__":
    test_1() 
